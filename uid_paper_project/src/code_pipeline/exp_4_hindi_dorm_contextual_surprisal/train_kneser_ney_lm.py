"""
 himanshu - revised - diagnostics added 
 comments also added
 
trigram Kneser-Ney language model

The training CSV is read twice:
 -  Count words and create the vocabulary.
 -  Replace rare words with <UNK> and build n-gram counts.

    order 2 -> bigram model
    order 3 -> trigram model

The saved model can later score a separate test corpus
"""

import argparse
import csv
import gzip
import math
import pickle
from collections import Counter
from pathlib import Path


DISCOUNT = 0.75
UNK_THRESHOLD = 1
PROGRESS_EVERY = 100_000
PROBABILITY_FLOOR = 1e-15
MODEL_VERSION = 1

START = "<s>"
END = "</s>"
UNK = "<UNK>"
SPECIAL_TOKENS = {START, END, UNK}


def read_sentences_from_csv(csv_path, encoding="utf-8-sig"):
 
    file_name = Path(csv_path)
    with file_name.open("r", encoding=encoding, newline="") as input_file:

        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:       # check for header row presence
            raise ValueError(f"CSV has no header row: {csv_path}")

        if "sentence" not in reader.fieldnames:
            raise ValueError(
                "The CSV must contain a column named 'sentence'. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            text = row["sentence"]
            tokens = text.strip().split()

            if tokens:
                yield tokens        # return one tokenized sentence at a time


class KneserNeyLanguageModel:

    """Bigram or trigram interpolated Kneser-Ney language model."""

    def __init__(self, order, discount=DISCOUNT, unk_threshold=UNK_THRESHOLD):
        if order not in (2, 3):
            raise ValueError("order must be 2 for bigram or 3 for trigram")
        if not 0 < discount < 1:
            raise ValueError("discount must be between 0 and 1")
        if unk_threshold < 0:
            raise ValueError("unk_threshold cannot be negative")

        self.order = order
        self.discount = discount
        self.unk_threshold = unk_threshold      # Frequency threshold for replacing rare words with <UNK>
        self.vocabulary = set()     # Set of retained words and special tokens

        # Counts obtained from the mapped training sentences.
        self.bigram_count = Counter()   # C(w1,w2) - Frequency of the bigram | for trigram : Frequency of the preceding bigram context (in first term of trigram formula)
        self.trigram_count = Counter()  # C(w1,w2,w3) - Trigram frequency (in first term of trigram formula)

        # Counts used by the bigram and unigram KN formulas.
        self.bigram_context_count = Counter()       # C(w1) - Total occurrences of words immediately after context w1, since context of w2 is w1
        self.bigram_unique_follower_count = Counter()       # N(w1,*) - Number of distinct words appearing after w1
        self.unique_predecessor_count = Counter()   # N(*,w2) - Number of distinct words appearing before w2, in trigram case: P_cont(w3)
        self.total_unique_bigram_types = 0  # Total number of distinct bigram types in the corpus, sigma (N(*,w))

        # Additional counts used by a trigram KN model.
        self.trigram_unique_follower_count = Counter()    # N(w1,w2,*) - Number of distinct words appearing after context (w1,w2)
        self.continuation_bigram_count = Counter()      # N(*,w2,w3) - no. of distinct words appearing immediately before the bigram (w2,w3) - C_cont (w2,w3)
        self.continuation_context_count = Counter()     # C_count(w2)  - Sum of continuation bigram counts for all followers of w2 (context of w3 in trigram formula - second term, P_KN(w3|w2))
        self.continuation_unique_follower_count = Counter() # N(w2,*) - Number of distinct followers having continuation bigram count after w2

        self.training_sentence_count = 0        # Total training sentences
        self.training_token_count = 0           # Total actual word tokens in training sentences
        self.unk_token_count = 0                # Number of training tokens replaced with <UNK>

    def get_start_tokens(self):
        """Return one <s> for bigram and two <s> tokens for trigram."""
        return [START] * (self.order - 1)

    def map_token(self, token):
        """Map a test word outside the training vocabulary to <UNK>."""
        if token in self.vocabulary:
            return token
        return UNK

    def normalize_tokens_for_inference(self, tokens):
        """Show the words that the trained model will actually score."""
        return [self.map_token(token) for token in tokens]

    def train_from_csv(
        self,
        csv_path,
        text_column="sentence",
        encoding="utf-8-sig",
        progress_every=PROGRESS_EVERY,
    ):
        """Train using two streaming passes through the large CSV."""

        raw_word_count = Counter()      # keys represent words and values represent their frequencies

        # First pass: Counts word frequencies and builds the vocabulary 
        print("Pass 1/2: counting word frequencies...")
        
        sentences = read_sentences_from_csv(csv_path,encoding)   # returns sentences - a generator to fetch one row of sentence at a time

        for sentence_number, tokens in enumerate(sentences, start = 1):
            # token frequencies are needed for vocabulary construction
            raw_word_count.update(tokens)       # ordinary unigram word counts - counts how many times each word occurs
            self.training_sentence_count = sentence_number
            self.training_token_count += len(tokens)        # Adds the number of tokens/words in the current sentence

            if progress_every and sentence_number % progress_every == 0:
                print(
                    f"  sentences={sentence_number:,}, "
                    f"raw vocabulary={len(raw_word_count):,}",      # no of unique keys  - words - vocab
                    flush=True,
                )

        if self.training_sentence_count == 0:
            raise ValueError("No non-empty sentences were found")

        kept_words = { word                 # set of unique words
            for word, count in raw_word_count.items()
            if count > self.unk_threshold
        }
        self.vocabulary = kept_words | SPECIAL_TOKENS

        del raw_word_count      # model no longer needs these raw unigram counts after constructing the vocabulary

        # The model now has a fixed vocabulary.
        # Any training word outside this vocabulary will be mapped to <UNK> during Pass 2

        print("Pass 2/2: building n-gram counts...")
        second_pass_sentence_count = 0

        sentences = read_sentences_from_csv(csv_path,encoding)   # returns new generator in sentence to fetch one row of sentence at a time

        for sentence_number, tokens in enumerate(sentences, start = 1):    

            second_pass_sentence_count = sentence_number
            mapped_tokens = [
                token if token in self.vocabulary else UNK
                for token in tokens
            ]

            self.unk_token_count += mapped_tokens.count(UNK)    # counts how many <UNK> tokens are present in the current mapped sentence

            sentence = self.get_start_tokens() + mapped_tokens + [END]

            for index in range(1, len(sentence)):
                bigram = (sentence[index - 1], sentence[index])
                self.bigram_count[bigram] += 1

            if self.order == 3:
                for index in range(2, len(sentence)):
                    trigram = (
                        sentence[index - 2],
                        sentence[index - 1],
                        sentence[index],
                    )
                    self.trigram_count[trigram] += 1

            if progress_every and sentence_number % progress_every == 0:
                print(
                    f"  sentences={sentence_number:,}, "
                    f"bigrams={len(self.bigram_count):,}, "
                    f"trigrams={len(self.trigram_count):,}",
                    flush=True,
                )

        if second_pass_sentence_count != self.training_sentence_count:
            raise RuntimeError(
                "The CSV changed between the first and second pass"
            )

        self.build_context_statistics()

    def build_context_statistics(self):
        """
            Derive the context and continuation counts used in KN smoothing.
            Convert the raw bigram and trigram counts collected during training 
            into the additional counts required by the KN probability formulas
        
        """

        self.bigram_context_count.clear()   #  C(w1,w2) - Frequency of the bigram | for trigram : Frequency of the preceding bigram context (in first term of trigram formula)
        self.bigram_unique_follower_count.clear()   # N(w1,*) - Number of distinct words appearing after w1
        self.unique_predecessor_count.clear() # N(*,w2) - Number of distinct words appearing before w2, in trigram case: P_cont(w3)

        for (previous_word, current_word), count in self.bigram_count.items():
            self.bigram_context_count[previous_word] += count
            self.bigram_unique_follower_count[previous_word] += 1
            self.unique_predecessor_count[current_word] += 1

        self.total_unique_bigram_types = len(self.bigram_count)     # Counts the total number of distinct bigram types stored in the training model.

        if self.order == 2:
            return

        self.trigram_unique_follower_count.clear() # N(w1,w2,*) - Number of distinct words appearing after context (w1,w2)
        self.continuation_bigram_count.clear()  # N(*,w2,w3) - no. of distinct words appearing immediately before the bigram (w2,w3) - C_cont (w2,w3)
        self.continuation_context_count.clear() # C_count(w2)  - Sum of continuation bigram counts for all followers of w2 (context of w3 in trigram formula - second term, P_KN(w3|w2))
        self.continuation_unique_follower_count.clear()   # N(w2,*) - Number of distinct followers having continuation bigram count after w2

        # Every distinct trigram (w1, w2, w3) adds one predecessor to (w2, w3).
        for word_1, word_2, word_3 in self.trigram_count:
            self.trigram_unique_follower_count[(word_1, word_2)] += 1
            self.continuation_bigram_count[(word_2, word_3)] += 1

        # Build lower-order continuation denominators
        for (previous_word, current_word), count in (self.continuation_bigram_count.items()):
            self.continuation_context_count[previous_word] += count
            self.continuation_unique_follower_count[previous_word] += 1

    def calculate_unigram_probability(self, word):              # P_count(w3) - in trigram KN, second term, in bigram continuation count formula P_KN(w3|w2)
        """
        Calculate KN continuation probability P_CONT(word)
        - receives the word for which we want a unigram continuation probability
        """

        word = self.map_token(word) # checks whether the word exists in the model vocabulary, If not, it replaces the word with <UNK>.

        # checks whether there are no distinct bigrams available -  fallback case.
        if self.total_unique_bigram_types == 0: 
            return 1.0 / max(len(self.vocabulary), 1)

        # P_cont(word) = unique count / unique bigram count
        return (                
            self.unique_predecessor_count.get(word, 0)
            / self.total_unique_bigram_types
        )

    def calculate_bigram_probability(self, previous_word, current_word):        # PKN (w3|w2) - in trigram formula- second term
        """Calculate an interpolated KN bigram probability of a word given its preceding word"""

        previous_word = self.map_token(previous_word)
        current_word = self.map_token(current_word)

        
        if self.order == 2:
            bigram_value = self.bigram_count.get((previous_word, current_word), 0)  # For a bigram model, bigram_value contains the observed frequency, C(w1, w2)
            context_value = self.bigram_context_count.get(previous_word, 0) # Gets the total number of times the context appears with any following word
            unique_followers = self.bigram_unique_follower_count.get(previous_word, 0)  # Gets the number of distinct words following the context.
        else:
            # Trigram KN uses continuation counts at its lower bigram level.
            # The bigram numerator comes from continuation bigram counts, rather than ordinary bigram frequencies.

            bigram_value = self.continuation_bigram_count.get((previous_word, current_word), 0)
            context_value = self.continuation_context_count.get(previous_word, 0)
            unique_followers = self.continuation_unique_follower_count.get(previous_word, 0)        # Obtains the number of distinct followers for the continuation-based context

        '''
        If the context has no usable count, the denominator would be zero.
        Instead of dividing by zero, the function directly returns the lower-order unigram continuation probability.
        '''

        if context_value == 0:
            return self.calculate_unigram_probability(current_word)

        first_term = (max(bigram_value - self.discount, 0) / context_value)     # max(..., 0) prevents the numerator from becoming negative.
        lambda_value = (self.discount * unique_followers  / context_value) # Calculates the interpolation weight, which represents the probability mass removed by discounting.

        return first_term + lambda_value * self.calculate_unigram_probability(current_word)

    def calculate_trigram_probability(self, word_1, word_2, word_3):
        """
        Calculate an interpolated KN trigram probability.
        P_KN( w3 | w1, w2)
        """

        word_1 = self.map_token(word_1)
        word_2 = self.map_token(word_2)
        word_3 = self.map_token(word_3)

        context = (word_1, word_2)          # Creates a tuple containing the two preceding words.
        context_value = self.bigram_count.get(context, 0)       # Retrieves C(w1,w2) - the denominator of the trigram formula

        if context_value == 0:
            return self.calculate_bigram_probability(word_2, word_3)

        trigram_value = self.trigram_count.get((word_1, word_2, word_3), 0)     # gets the count C(w1,w2,w3)
        unique_followers = self.trigram_unique_follower_count.get(context, 0)   # Gets the number of distinct words that follow (w1,w2).

        first_term = (max(trigram_value - self.discount, 0) / context_value)    # Calculates the discounted trigram contribution
        lambda_value = (self.discount * unique_followers / context_value)   # Calculates the trigram interpolation weight

        return first_term + lambda_value * self.calculate_bigram_probability(word_2, word_3)

    def compute_probability_with_diagnostics(self, context_words, current_word):
        """Calculate one event probability and report its backoff level."""

        expected_context_length = self.order - 1        # order = 3, the expected context length is 2
        if len(context_words) != expected_context_length:
            raise ValueError(
                f"Order {self.order} requires "
                f"{expected_context_length} context word(s)"
            )

        original_context = list(context_words)      # Stores the original context words before replacing any unknown words
        scored_context = [self.map_token(word) for word in context_words]   # Creates the context actually used by the model, replacing out-of-vocabulary words with <UNK>
        scored_word = self.map_token(current_word)          # Maps the current word in the same way

        if self.order == 2:
            previous_word = scored_context[0]       # Extracts the single preceding word
            context_seen = self.bigram_context_count.get(previous_word, 0) > 0      # Checks whether the context occurred during training
            ngram_seen = self.bigram_count.get((previous_word, scored_word), 0) > 0     # Checks whether the complete bigram has been observed.

            # If the context exists, calculate the interpolated bigram probability and label the level bigram
            # If the context does not exist, use the unigram continuation probability.
            if context_seen:
                probability = self.calculate_bigram_probability(previous_word, scored_word)
                backoff_level = "bigram"
            else:
                probability = self.calculate_unigram_probability(scored_word)
                backoff_level = "unigram"
        else:
            word_1, word_2 = scored_context
            # Checks whether the preceding two-word context exists
            context_seen = self.bigram_count.get((word_1, word_2), 0) > 0       # This checks whether the context occurred in training
             # checks for existence of full tri-gram
            ngram_seen = self.trigram_count.get((word_1, word_2, scored_word), 0) > 0

            if context_seen:    # The two-word context exists, so calculate the interpolated trigram probability
                probability = self.calculate_trigram_probability(word_1, word_2, scored_word)
                backoff_level = "trigram"
            # The trigram context does not exist, but the lower-order bigram context has usable continuation statistics    
            elif self.continuation_context_count.get(word_2, 0) > 0:
                probability = self.calculate_bigram_probability(word_2, scored_word)
                backoff_level = "bigram"
            # Neither of the higher-order contexts is available, so use the unigram continuation probability.
            else:
                probability = self.calculate_unigram_probability(scored_word)
                backoff_level = "unigram"

        final_probability = max(probability, PROBABILITY_FLOOR)

        return {
            "prob": float(final_probability),
            "backoff_level": backoff_level,
            "highest_order_context_seen": context_seen,
            "highest_order_ngram_seen": ngram_seen,
            "original_word_seen_in_model_vocab": current_word in self.vocabulary,
            "used_unk_for_current_word": (
                scored_word == UNK and current_word != UNK
            ),
            "used_floor_probability": final_probability > probability,
            "original_context": original_context,
            "scored_context": scored_context,
            "scored_current_word": scored_word,
        }

    def compute_probability(self, context_words, current_word):
        """Return only the probability for one bigram/trigram event."""
        details = self.compute_probability_with_diagnostics(context_words, current_word)
        return details["prob"]

    def score_sentence_tokens(self, tokens):
        
        """Score real words """

        tokens = list(tokens)
        start_tokens = self.get_start_tokens()      #  return [START] * (self.order - 1)
        sentence = start_tokens + tokens        # for order = 3, start_tokens = ["<s>", "<s>"]
        token_records = []      # Creates an empty list to hold the probability, surprisal and diagnostic information for every real word

        for token_index in range(len(tokens)):
            sentence_index = len(start_tokens) + token_index
             # eg context_words = sentence[2 - 2:2] -> context_words = sentence[0:2] - context_words = ["<s>", "<s>"]
            context_words = sentence[sentence_index - len(start_tokens):sentence_index]
            current_word = sentence[sentence_index]

            details = self.compute_probability_with_diagnostics(context_words, current_word)

            token_records.append({
                "token_index": token_index + 1,
                "previous_context_2": (context_words[-2] if self.order == 3 else None),
                "previous_context_1": context_words[-1],
                "current_word": current_word,
                "scored_current_word": details["scored_current_word"],
                "prob": details["prob"],
                "surprisal_bits": -math.log2(details["prob"]),
                "backoff_level": details["backoff_level"],
                "highest_order_context_seen": details["highest_order_context_seen"],
                "highest_order_ngram_seen": details["highest_order_ngram_seen"],
                "used_unk_for_current_word": details["used_unk_for_current_word"],
                "used_floor_probability": details["used_floor_probability"],
            })

        return token_records

    def sentence_token_surprisals(self, tokens):
        """Return the contextual-surprisal records """
        return self.score_sentence_tokens(tokens)

    def score_end_token(self, tokens):
        """Score </s> separately from the real-word surprisal vector."""

        sentence = self.get_start_tokens() + list(tokens) + [END]
        context_words = sentence[-self.order:-1]

        return self.compute_probability_with_diagnostics(context_words, END)

    def sentence_logprob(self, tokens, include_end_token=True):
        """Return base-2 sentence log probability."""

        token_records = self.score_sentence_tokens(tokens)
        total_logprob = sum(
            math.log2(record["prob"])
            for record in token_records
        )

        if include_end_token:
            total_logprob += math.log2(self.score_end_token(tokens)["prob"])

        return float(total_logprob)
    
    # this function used for experiment
    def score_sentence_with_diagnostics(self, tokens):
        """Return token surprisals, sentence scores, and backoff counts."""

        token_records = self.score_sentence_tokens(tokens)
        word_logprob = sum(
            math.log2(record["prob"])
            for record in token_records
        )
        end_logprob = math.log2(self.score_end_token(tokens)["prob"])
        token_count = len(token_records)
        highest_level = "bigram" if self.order == 2 else "trigram"

        average_surprisal = 0.0
        if token_count:
            average_surprisal = -word_logprob / token_count

        return {
            "model_order": self.order,
            "sentence_lexical_logprob": float(word_logprob + end_logprob),
            "sentence_word_logprob": float(word_logprob),
            "average_token_lexical_surprisal": float(average_surprisal),
            "token_surprisal_records": token_records,
            "real_token_count": token_count,
            "unseen_token_count": sum(
                row["used_unk_for_current_word"] for row in token_records
            ),
            "unseen_highest_order_ngram_count": sum(
                not row["highest_order_ngram_seen"] for row in token_records
            ),
            "highest_order_backoff_count": sum(
                row["backoff_level"] != highest_level
                for row in token_records
            ),
            "trigram_level_count": sum(
                row["backoff_level"] == "trigram"
                for row in token_records
            ),
            "bigram_level_count": sum(
                row["backoff_level"] == "bigram"
                for row in token_records
            ),
            "unigram_level_count": sum(
                row["backoff_level"] == "unigram"
                for row in token_records
            ),
            "floor_probability_count": sum(
                row["used_floor_probability"] for row in token_records
            ),
        }

    def save_model(self, model_path):
        """Save all model variables in a compressed pickle file."""

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # vars(self)  - built-in Python function - 
        # returns a dictionary containing all attributes currently stored in calling object of class  
        model_data = {
            "model_version": MODEL_VERSION,
            "state": vars(self),
        }
        '''
        Serialization means converting a Python object 
        (such as a dictionary, list, Counter, or trained model) 
        into a sequence of bytes so that it can be saved in a file and loaded again later.
        '''
        with gzip.open(model_path, "wb", compresslevel=6) as output_file:   # allows the serialized data to be written in compressed form
            pickle.dump(        # serializes the dictionary and its contents into bytes and writes them to the file.
                model_data,
                output_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def training_summary(self):
        """Return the main training and model-size values."""

        return {
            "model_order": self.order,
            "training_sentence_count": self.training_sentence_count,
            "training_token_count": self.training_token_count,
            "vocabulary_size": len(self.vocabulary - SPECIAL_TOKENS),
            "unk_token_count": self.unk_token_count,
            "unique_bigram_count": len(self.bigram_count),
            "unique_trigram_count": len(self.trigram_count),
        }


def load_model(model_path):
    """ load a model created by save_model()  
    opens the saved compressed file and restores the dictionary that was previously stored
    eg
    model_data = 
    {
    "model_version": 1,
    "state": {
        "order": 2,
        "discount": 0.75,
        "vocabulary": {...},
        "bigram_count": Counter(...),
        ...
    }
}
    """

    with gzip.open(model_path, "rb") as input_file:
        model_data = pickle.load(input_file)        # reverse operation is called deserialization

    if model_data.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported model file version")

    state = model_data["state"]
    model = KneserNeyLanguageModel(             # essentially an empty model as of now
        order=state["order"],
        discount=state["discount"],
        unk_threshold=state["unk_threshold"],
    )
    vars(model).update(state)           # fetches all  the data/variables and put them back into the newly created model object
    return model


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description="Train a bigram or trigram Kneser-Ney model from CSV"
    )
    parser.add_argument("--input_csv", type = Path, required = True)
    parser.add_argument("--output_model", type = Path, required = True)
    parser.add_argument("--order", type =  int, choices = (2, 3), required = True)
   
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    model = KneserNeyLanguageModel(order = args.order, discount = DISCOUNT, unk_threshold = UNK_THRESHOLD)

    model.train_from_csv(
        csv_path = args.input_csv,
        text_column = "sentence",
        encoding = "utf-8-sig",
        progress_every = PROGRESS_EVERY,
    )
    model.save_model(args.output_model)

    print("\nTraining complete.")
    for name, value in model.training_summary().items():
        print(f"{name}: {value:,}") 
        #print(f"{name}: {value:,}" if isinstance(value, int) else f"{name}: {value}")
    print(f"model_saved_at: {args.output_model}")


if __name__ == "__main__":
    main()
