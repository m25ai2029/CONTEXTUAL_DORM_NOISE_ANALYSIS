"""
himanshu - revised - added more diagnostic results
this file generate 4 Hindi sentence versions using contextual trigram surprisal:
Actual, Uniform, Asymmetric, and Random

- loads a previously trained Kneser-Ney language mode - Contextual
surprisal is recalculated for every distinct proposed word order

Uniform version:  Repeatedly try adjacent swaps and keep a swap only when DORM decreases
Asymmetric version:
    Repeatedly try a 2 word swap and keep it only when the information centre
    moves toward the beginning of the sentence

"""

import argparse
import csv
import json
import random
import re
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from train_kneser_ney_lm import (KneserNeyLanguageModel, load_model)

LM_MODEL_PATH = Path("./outputs/mtp_2_features/exp_4_demo_output/trigram_kneser_ney_model.pkl.gz")

RANDOM_SEED = 42
ASYMMETRIC_SEED = 42
UNIFORM_MAX_EVALUATIONS = 200
ASYMMETRIC_ITERATIONS = 100
PROGRESS_EVERY = 10_000
IMPROVEMENT_TOLERANCE = 1e-12

EDGE_PUNCTUATION_RE = re.compile(
    r"^[\s\u0964\u0965|.,!?;:'\"“”‘’()\[\]{}<>…—–\-]+|"
    r"[\s\u0964\u0965|.,!?;:'\"“”‘’()\[\]{}<>…—–\-]+$"
)


@dataclass(frozen=True)
class TokenItem:
    word: str
    original_index: int


def clean_token_edges(token: str) -> str:
    """ removes punctuation only from the beginning and end of a token """
    return EDGE_PUNCTUATION_RE.sub("", str(token)).strip()


def tokenize_hindi_sentence(sentence: str) -> List[str]:
    """ whitespace tokenize an already-clean Hindi sentence 
    just a sanity check 
    """
    tokens = []

    for raw_token in str(sentence).split():
        token = clean_token_edges(raw_token)
        if token:
            tokens.append(token)

    return tokens


def make_token_items(tokens: Sequence[str]) -> List[TokenItem]:
    """Attach each token to its position in the actual sentence'
    returns list of TokenItems - [TokenItem(word="ram", original_index=0), TokenItem(word="ghar", original_index=1),...]
    """
    items = []

    for index, token in enumerate(tokens):
        items.append(
            TokenItem(
                word = token,
                original_index = index,
            )
        )

    return items


def words_from_items(items: Sequence[TokenItem]) -> List[str]:
    """Extract the surface-word sequence from TokenItem objects."""
    return [item.word for item in items]


def sentence_from_items(items: Sequence[TokenItem]) -> str:
    """Join a TokenItem sequence into a sentence string."""
    return " ".join(words_from_items(items))

def score_sentence_order(
    items: Sequence[TokenItem],
    model,
    score_cache: Dict[Tuple[str, ...], List[dict]],
) -> List[dict]:
    
    """
    Score one complete word order
    A distinct word order is scored only once within the current sentence
  
    """
    words = words_from_items(items)     # words = ["ram", "ghar", "gaya"] - return [item.word for item in items]
    cache_key = tuple(words)    # creates an immutable dictionary key - ("ram", "ghar", "gaya")
    # A list cannot normally be used as a dictionary key, also preserves order
    if cache_key not in score_cache:            # checks whether this exact word order has already been scored
        # If it has not been scored, the language model calculates each word’s contextual surprisal
        score_cache[cache_key] = model.sentence_token_surprisals(words)

    return score_cache[cache_key]


def surprisal_values_from_records(token_records: Sequence[dict]) -> List[float]:
    """Extract real-word contextual surprisal values in bits."""
    return [float(record["surprisal_bits"]) for record in token_records]


def uidglob_from_surprisal_values(surprisal_values: Sequence[float]) -> float:
    """Compute UIDglob: negative population variance of token surprisal."""
    number_of_tokens = len(surprisal_values)

    if number_of_tokens == 0:
        return 0.0

    mean_surprisal = sum(surprisal_values) / number_of_tokens

    squared_deviations = 0.0
    for value in surprisal_values:
        squared_deviations += (value - mean_surprisal) ** 2

    return -squared_deviations / number_of_tokens


def uidloc_from_surprisal_values(surprisal_values: Sequence[float]) -> float:
    """Compute UIDloc: negative local squared change, divided by N."""
    number_of_tokens = len(surprisal_values)

    if number_of_tokens <= 1:
        return 0.0

    squared_changes = 0.0

    for index in range(1, number_of_tokens):
        difference = surprisal_values[index] - surprisal_values[index - 1]
        squared_changes += difference ** 2

    return -squared_changes / number_of_tokens


def dorm_from_surprisal_values(surprisal_values: Sequence[float]) -> float:

    """
    this func computes DORM as the sample variance of adjacent-pair means
    """
    number_of_tokens = len(surprisal_values)

    if number_of_tokens <= 2:
        return 0.0

    pair_means = []

    for index in range(number_of_tokens - 1):
        pair_mean = ( surprisal_values[index] + surprisal_values[index + 1] ) / 2.0
        pair_means.append(pair_mean)

    mean_of_pair_means = sum(pair_means) / len(pair_means)

    squared_deviations = 0.0
    for pair_mean in pair_means:
        squared_deviations += (pair_mean - mean_of_pair_means) ** 2

    return squared_deviations / (len(pair_means) - 1)


def information_center_from_surprisal_values(
    surprisal_values: Sequence[float],
) -> float:
    
    """
    this function returns the surprisal-weighted mean token position- 
    Positions begin at 1. A smaller value places more information toward the
    beginning of the sentence
    """
    total_surprisal = sum(surprisal_values)

    if total_surprisal == 0:
        return 0.0

    weighted_position_sum = 0.0

    for position, surprisal in enumerate(surprisal_values, start=1):
        weighted_position_sum += position * surprisal

    return weighted_position_sum / total_surprisal




def initial_uniform_proposal(
    actual_items: Sequence[TokenItem],
    actual_records: Sequence[dict],
) -> List[TokenItem]:
    
    """
    
    uses the current contextual surprisal values from the actual sentence only to decide how to rearrange the words initially
    The proposal is rescored after reordering; the old surprisal values are not
    used to calculate the proposal's DORM.
    - the new order must be rescored with the language model because contextual surprisal changes when word order changes.
    """
    item_surprisal_pairs = list(zip(actual_items, actual_records))
    item_surprisal_pairs.sort(
        key=lambda pair: (
            float(pair[1]["surprisal_bits"]),
            -pair[0].original_index,        # tie-breaker. If two words have exactly the same surprisal, the word occurring earlier in the original sentence is placed first.
        ),
        reverse=True,
    )

    proposal_items = [pair[0] for pair in item_surprisal_pairs]

    left = 0
    right = len(proposal_items) - 1
    # form low-high-low-high-like pattern
    while right > left:
        proposal_items[left], proposal_items[right] = (
            proposal_items[right],
            proposal_items[left],
        )
        left += 2
        right -= 2

    return proposal_items


def initial_asymmetric_proposal(
    actual_items: Sequence[TokenItem],
    actual_records: Sequence[dict],
) -> List[TokenItem]:
    
    """
    Propose high-current-surprisal words first, then rescore the new order.
    """
    item_surprisal_pairs = list(zip(actual_items, actual_records))
    item_surprisal_pairs.sort(
        key=lambda pair: (
            float(pair[1]["surprisal_bits"]),
            -pair[0].original_index,
        ),
        reverse=True,
    )

    return [pair[0] for pair in item_surprisal_pairs]


def optimize_uniform_order(
    actual_items: Sequence[TokenItem],
    model,
    score_cache: Dict[Tuple[str, ...], List[dict]],
    max_evaluations: int,
) -> Tuple[List[TokenItem], int, int]:
    
    """
    This function finds a context sensitive uniform order using adjacent-swap 

    Every proposed complete order is rescored. A proposal is accepted only
    when its contextual-surprisal DORM is strictly smaller.
    """
    '''
    actual_items is list of TokenItem data object, actual_items = [ 
                                                            TokenItem("ram",0)
                                                            TokenItem("ghar",1)
                                                            ]         

    actual_records is  list of dictionaries , for all words in a sentence
    There is one dictionary for each real word in the sentence
    example- 
    actual_records = [
    {
        "token_index": 1,
        "previous_context_2": "<s>",
        "previous_context_1": "<s>",
        "current_word": "ram",
        "scored_current_word": "ram",
        "prob": 0.3728125,
        "surprisal_bits": 1.4235,
        "scoring_level": "trigram",
        "highest_order_context_seen": True,
        "highest_order_ngram_seen": True,

    },
    {
        "token_index": 2,
        "previous_context_2": "<s>",
        "previous_context_1": "ram",
        "current_word": "ghar",
        "scored_current_word": "ghar",
        "prob": 0.46,
        "surprisal_bits": 1.1203,
        "socring_level": "trigram",
        "highest_order_context_seen": True,
        "highest_order_ngram_seen": True,
    },
 ]
    ''' 

   

    actual_items = list(actual_items)
    
    '''
    below line computes the contextual surprisal of every word in the actual sentence order and 
    stores the result in actual_records
    '''
    actual_records = score_sentence_order(actual_items, model, score_cache)
    actual_values = surprisal_values_from_records(actual_records)   # list of all float surprisal values of all tokens
    actual_dorm = dorm_from_surprisal_values(actual_values)

    proposal_items = initial_uniform_proposal( actual_items, actual_records)
    proposal_records = score_sentence_order( proposal_items, model, score_cache)
    proposal_values = surprisal_values_from_records(proposal_records)
    proposal_dorm = dorm_from_surprisal_values(proposal_values)

    evaluation_count = 1
    accepted_swap_count = 0 # number of accepted improvements

    if proposal_dorm < actual_dorm - IMPROVEMENT_TOLERANCE:     # improvement should be meaningful, not tiny diff
        current_items = proposal_items
        current_dorm = proposal_dorm
        accepted_swap_count = 1
    else:
        current_items = actual_items
        current_dorm = actual_dorm

    index = 0

    while (
        index + 1 < len(current_items)
        and evaluation_count < max_evaluations
    ):
        candidate_items = list(current_items)
        candidate_items[index], candidate_items[index + 1] = (
            candidate_items[index + 1],
            candidate_items[index],
        )

        candidate_records = score_sentence_order(candidate_items, model, score_cache)
        candidate_values = surprisal_values_from_records(candidate_records)
        candidate_dorm = dorm_from_surprisal_values(candidate_values)
        evaluation_count += 1

        if candidate_dorm < current_dorm - IMPROVEMENT_TOLERANCE:
            current_items = candidate_items
            current_dorm = candidate_dorm
            accepted_swap_count += 1
            index = 0
        else:
            index += 1

    return current_items, evaluation_count, accepted_swap_count


def optimize_asymmetric_order(
    actual_items: Sequence[TokenItem],
    model,
    score_cache: Dict[Tuple[str, ...], List[dict]],
    rng: random.Random,
    max_iterations: int,
) -> Tuple[List[TokenItem], int, int]:
    
    """
    essentually , it moves contextual surprisal toward the beginning of the sentence

    At each iteration, two positions are selected -  The complete candidate order
    is rescored and retained only if its information centre becomes smaller
    """
    actual_items = list(actual_items)

    if len(actual_items) <= 1:
        return actual_items, 0, 0

    actual_records = score_sentence_order(actual_items, model, score_cache)
    actual_values = surprisal_values_from_records(actual_records)
    actual_center = information_center_from_surprisal_values(actual_values)

    proposal_items = initial_asymmetric_proposal(actual_items, actual_records)
    proposal_records = score_sentence_order(proposal_items,model,score_cache)
    proposal_values = surprisal_values_from_records(proposal_records)
    proposal_center = information_center_from_surprisal_values(proposal_values)

    evaluation_count = 1
    accepted_swap_count = 0

    if proposal_center < actual_center - IMPROVEMENT_TOLERANCE:
        current_items = proposal_items
        current_center = proposal_center
        accepted_swap_count = 1
    else:
        current_items = actual_items
        current_center = actual_center

    for _ in range(max_iterations):
        first_index, second_index = rng.sample(range(len(current_items)), 2)

        candidate_items = list(current_items)
        candidate_items[first_index], candidate_items[second_index] = (
            candidate_items[second_index],
            candidate_items[first_index],
        )

        candidate_records = score_sentence_order(candidate_items, model, score_cache)
        candidate_values = surprisal_values_from_records(candidate_records)
        candidate_center = information_center_from_surprisal_values(candidate_values)
        evaluation_count += 1

        if candidate_center < current_center - IMPROVEMENT_TOLERANCE:
            current_items = candidate_items
            current_center = candidate_center
            accepted_swap_count += 1

    return current_items, evaluation_count, accepted_swap_count


def create_random_order(
    actual_items: Sequence[TokenItem],
    rng: random.Random,
) -> List[TokenItem]:
    
    """create one reproducible random permutation of the actual words."""
    random_items = list(actual_items)
    rng.shuffle(random_items)
    return random_items


def token_records_to_json(
    items: Sequence[TokenItem],
    token_records: Sequence[dict],
    model,
) -> str:
    """
    Convert model token records to JSON.

    """
    output_records = []

    for item, record in zip(items, token_records):
        surprisal_bits = float(record["surprisal_bits"])
        scoring_level = record.get(
            "scoring_level",
            record.get("probability_level", record.get("backoff_level")),
        )

        scored_current_word = model.map_token(item.word)
        previous_context_1 = record.get("previous_context_1")
        scored_previous_context_1 = model.map_token(previous_context_1)

        if scoring_level == "trigram":
            previous_context_2 = record.get("previous_context_2")
            scored_previous_context_2 = model.map_token(previous_context_2)
            scoring_level_ngram_seen = model.trigram_count.get(
                (
                    scored_previous_context_2,
                    scored_previous_context_1,
                    scored_current_word,
                ),
                0,
            ) > 0
        elif scoring_level == "bigram":
            if model.order == 3:
                scoring_level_ngram_seen = model.continuation_bigram_count.get(
                    (scored_previous_context_1, scored_current_word),
                    0,
                ) > 0
            else:
                scoring_level_ngram_seen = model.bigram_count.get(
                    (scored_previous_context_1, scored_current_word),
                    0,
                ) > 0
        elif scoring_level == "unigram":
            scoring_level_ngram_seen = model.unique_predecessor_count.get(
                scored_current_word,
                0,
            ) > 0
        else:
            scoring_level_ngram_seen = False

        output_records.append(
            {
                "token_index": int(record["token_index"]),
                "token": item.word,
                "information_content": surprisal_bits,
                "original_index": int(item.original_index),
                "previous_context_2": record.get("previous_context_2"),
                "previous_context_1": record.get("previous_context_1"),
                "prob": float(record["prob"]),
                "surprisal_bits": surprisal_bits,
                "scoring_level": scoring_level,
                "highest_order_context_seen": bool(
                    record["highest_order_context_seen"]
                ),
                "highest_order_ngram_seen": bool(
                    record["highest_order_ngram_seen"]
                ),
                "scoring_level_ngram_seen": bool(scoring_level_ngram_seen),
                "used_floor_probability": bool(
                    record["used_floor_probability"]
                ),
            }
        )

    return json.dumps(output_records, ensure_ascii=False)


def build_output_row(
    sent_count: str,
    version_type: str,
    items: Sequence[TokenItem],
    model,
    score_cache: Dict[Tuple[str, ...], List[dict]],
    optimization_evaluations: int,
    accepted_swaps: int,
) -> dict:
    
    """Build one finalized output row without rescoring the same order twice."""

    token_records = score_sentence_order(items, model, score_cache)
    surprisal_values = surprisal_values_from_records(token_records)

    total_bits = float(sum(surprisal_values))
    token_count = len(surprisal_values)
    #average_surprisal = total_bits / token_count if token_count else 0.0
    return {
        "sent_count": sent_count,
        "type": version_type,
        "sentence": sentence_from_items(items),
 #       "UIDGlob": uidglob_from_surprisal_values(surprisal_values),
 #       "UIDloc": uidloc_from_surprisal_values(surprisal_values),
        "DORM": dorm_from_surprisal_values(surprisal_values),
        "token_ic_json": token_records_to_json(items, token_records, model),
        "total_bits": total_bits,
    }


def generate_four_versions_for_sentence(
    sent_count: str,
    sentence: str,
    model,
    random_order_rng: random.Random,
    asymmetric_rng: random.Random,
    uniform_max_evaluations: int,
    asymmetric_iterations: int,
) -> List[dict]:
    
    """this is main function which generates and scores the 4 contextual-surprisal sentence versions"""

    tokens = tokenize_hindi_sentence(sentence)

    if not tokens:
        return []

    actual_items = make_token_items(tokens)
    score_cache: Dict[Tuple[str, ...], List[dict]] = {}

    uniform_items, uniform_evaluations, uniform_accepted = (
        optimize_uniform_order(
            actual_items = actual_items,
            model = model,
            score_cache = score_cache,
            max_evaluations = uniform_max_evaluations,
        )
    )

    asymmetric_items, asymmetric_evaluations, asymmetric_accepted = (
        optimize_asymmetric_order(
            actual_items = actual_items,
            model = model,
            score_cache = score_cache,
            rng = asymmetric_rng,
            max_iterations = asymmetric_iterations,
        )
    )

    random_items = create_random_order(actual_items, random_order_rng)

    actual_row = build_output_row(
    sent_count=sent_count,
    version_type="Actual",
    items=actual_items,
    model=model,
    score_cache=score_cache,
    optimization_evaluations=0,
    accepted_swaps=0,
    )   

    uniform_row = build_output_row(
    sent_count=sent_count,
    version_type="Uniform",
    items=uniform_items,
    model=model,
    score_cache=score_cache,
    optimization_evaluations=uniform_evaluations,
    accepted_swaps=uniform_accepted,
    )

    asymmetric_row = build_output_row(
    sent_count=sent_count,
    version_type="Asymmetric",
    items=asymmetric_items,
    model=model,
    score_cache=score_cache,
    optimization_evaluations=asymmetric_evaluations,
    accepted_swaps=asymmetric_accepted,
    )

    random_row = build_output_row(
    sent_count=sent_count,
    version_type="Random",
    items=random_items,
    model=model,
    score_cache=score_cache,
    optimization_evaluations=0,
    accepted_swaps=0,
    )

    output_rows = [actual_row, uniform_row, asymmetric_row, random_row]

    return output_rows



def generate_output_file(
    input_csv: Path,
    output_csv: Path,
    model,
    random_seed: int,
    asymmetric_seed: int,
    uniform_max_evaluations: int,
    asymmetric_iterations: int,
    progress_every: int,
) -> Tuple[int, int, int]:
    
    """ this function reads one input sentence at a time and immediately write four rows"""
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    random_order_rng = random.Random(random_seed)
    asymmetric_rng = random.Random(asymmetric_seed)

    fieldnames = [
        "sent_count",
        "type",
        "sentence",
        "DORM",
        "token_ic_json",
        "total_bits",
    ]

    input_sentence_count = 0
    output_row_count = 0
    skipped_sentence_count = 0

    with (
        input_csv.open("r", encoding="utf-8-sig", newline="") as input_file, 
        output_csv.open("w", encoding="utf-8-sig", newline="") as output_file,
    ):
        '''
        DictReader automatically sets up each row as a dictionary 
        where the keys are the column headers and the values are the cells in that row
        reader or writer - iterator object
        '''
        reader = csv.DictReader(input_file, delimiter=',')

        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header row: {input_csv}")

        reader.fieldnames = [column.strip() for column in reader.fieldnames]

        required_columns = {"sent_count", "sentence"}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"Input file must contain {sorted(required_columns)}. "
                f"Missing: {sorted(missing_columns)}. "
                f"Found: {reader.fieldnames}"
            )

        writer = csv.DictWriter(output_file, fieldnames = fieldnames)
        writer.writeheader()

        for row in reader:
            input_sentence_count += 1

            sent_count = str(row.get("sent_count", "")).strip()
            sentence = str(row.get("sentence", "")).strip()

            output_rows = generate_four_versions_for_sentence(
                sent_count=sent_count,
                sentence=sentence,
                model=model,
                random_order_rng=random_order_rng,
                asymmetric_rng=asymmetric_rng,
                uniform_max_evaluations=uniform_max_evaluations,
                asymmetric_iterations=asymmetric_iterations,
            )

            if not output_rows:
                skipped_sentence_count += 1
                continue

            writer.writerows(output_rows)
            output_row_count += len(output_rows)

            if (progress_every and input_sentence_count % progress_every == 0):
                print(
                    f"Processed sentences={input_sentence_count:,}, "
                    f"written rows={output_row_count:,}",
                    flush=True,
                )

    return input_sentence_count, output_row_count, skipped_sentence_count


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Actual, Uniform, Asymmetric, and Random Hindi sentence "
            "versions using contextual Kneser-Ney surprisal"
        )
    )

    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--order", type=int, required=True)

    return parser.parse_args()

def load_language_model(model_path: Path) -> KneserNeyLanguageModel:
    """  load and validate the trained bigram or trigram model """

    if not model_path.exists():
        raise FileNotFoundError(f"Lexical model file not found: {model_path}")

    model = load_model(model_path)  # calls function from train_kneser_ney_lm.py file

    if model.order not in (2, 3):           # a check only
        raise ValueError("The loaded language model must have order 2 or 3.")

    return model



def main():
    args = parse_arguments()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if args.order == 2:
        LM_MODEL_PATH = LM_MODEL_PATH = Path("./outputs/mtp_2_features/exp_4_demo_output/bigram_kneser_ney_model.pkl.gz")
    elif args.order == 3:
        LM_MODEL_PATH = LM_MODEL_PATH = Path("./outputs/mtp_2_features/exp_4_demo_output/trigram_kneser_ney_model.pkl.gz")
    else:
        raise ValueError("Incorrect Value of Order entered, must be either 2 or 3")
    
    
    print("Loading trained language model...")
    language_model = load_language_model(LM_MODEL_PATH)
    
    print(f"Loaded order - {language_model.order} Kneser-Ney language model")

    print("Generating four contextual-surprisal versions...")

    input_count, output_count, skipped_count = generate_output_file(
        input_csv = args.input_csv,
        output_csv = args.output_csv,
        model = language_model,
        random_seed = RANDOM_SEED,
        asymmetric_seed = ASYMMETRIC_SEED,
        uniform_max_evaluations = UNIFORM_MAX_EVALUATIONS,
        asymmetric_iterations = ASYMMETRIC_ITERATIONS,
        progress_every = PROGRESS_EVERY,
    )

    print("  Contextual Version Generation Complete ")
    print(f"Model Path                  : {LM_MODEL_PATH}")
    print(f"Model order                 : {language_model.order}")
    print(f"No. of Input sentences      : {input_count:,}")
    print(f"Output rows                 : {output_count:,}")
    print(f"Output file                 : {args.output_csv}")


if __name__ == "__main__":
    main()
