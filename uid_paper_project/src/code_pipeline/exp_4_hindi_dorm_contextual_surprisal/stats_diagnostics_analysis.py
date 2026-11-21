#!/usr/bin/env python3
"""
himanshu - ver 2.0 of this file

this file performs follwing: 
 - Reads the four-version CSV in chunks and summarizes by sentence version: 
   -- total contextual surprisal
   --  average token surprisal
   -- maximum single-token share of total bits
   -- share held by the top k tokens, where k =3 of sentence length
   -- maximum share in any contiguous k-token window
   -- share in the first k positions
 
 revised 

"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

NOISE_FRACTION = 0.30
CATASTROPHIC_THRESHOLD = 0.50
CHUNK_SIZE = 50_000
PROGRESS_EVERY = 100_000
VERSION_ORDER = ["Uniform", "Actual", "Random", "Asymmetric"]


def deletion_count(n: int, fraction: float = NOISE_FRACTION) -> int:
    if n <= 0:
        return 0
    return min(n, max(1, int(round(n * fraction))))


def parse_records(text: str) -> List[dict]:
    records = json.loads(str(text))
    if not isinstance(records, list):
        raise ValueError("token_ic_json must contain a JSON list")
    return records


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def information_center(values: List[float]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    return sum((i + 1) * v for i, v in enumerate(values)) / total


def max_contiguous_share(values: List[float], k: int) -> float:

    '''
    this function calculates  -  which group among all possible consecutive groups of k=3 words, 
    contains the largest proportion of sentence information
    '''
    total = sum(values)     # total sentence information
    if total <= 0 or k <= 0:
        return 0.0
    if k >= len(values):         # if the deletion window includes the whole sentence, then 100% of the information is removed
        return 1.0
    
    window = sum(values[:k])        # first contiguous window
    best = window                   # initially first window is considered the best
    for i in range(k, len(values)):         # sliding window, instead of recalculating 3 values each time, adds new token, remove old token
        window += values[i] - values[i-k]
        if window > best:
            best = window           # new largest window
    return best / total             # share of total info for that max window


def random_k_catastrophic_probability(
    shares: List[float],
    k: int,
    threshold: float = CATASTROPHIC_THRESHOLD,
) -> float:
    
    """
    this function calculates exact probability that k=3 uniformly random deleted positions lose at least
    threshold of the sentence information - that is deleted words contain >= 50% sentence info
    For our  10-token,  k=3 means there are  C(10, 3)=120 equally likely position sets

    """
    n = len(shares)
    k = 3
    """
    if n == 0 or k <= 0:
        return 0.0
    if k >= n:
        return float(sum(shares) >= threshold)

    """

    '''
     we will take the 3 most informative words possible. If even they contain less than 50%, 
     then no combination of 3 words could ever reach 50% - first check
    '''
    sorted_desc = sorted(shares, reverse=True)
    if sum(sorted_desc[:k]) < threshold:
        return 0.0

    total_combinations = math.comb(n, k)

    '''
    Tried to make generic, but have to test for k = 1 and k = 2 condition - can be researched later

    if k == 1:
        catastrophic_count = sum(1 for value in shares if value >= threshold)
        return catastrophic_count / total_combinations

    sorted_asc = sorted(shares)

    if k == 2:
        catastrophic_count = 0
        left = 0
        right = n - 1

        while left < right:
            if sorted_asc[left] + sorted_asc[right] >= threshold:
                catastrophic_count += right - left
                right -= 1
            else:
                left += 1

        return catastrophic_count / total_combinations
    '''

    '''
    we fix one token -  use left + right to find the other two, if first + left + right ≥ threshold:
    every value between left and right also works with that right value
    now, count several combinations at once - move right leftward
    Else:  the left value is too small  - move left rightward
    
    '''
    sorted_asc = sorted(shares)
    if k == 3:
        catastrophic_count = 0

        for first_index in range(n - 2):
            left = first_index + 1
            right = n - 1               # 2-pointer technique

            while left < right:
                current_share = (sorted_asc[first_index] + sorted_asc[left] + sorted_asc[right])

                if current_share >= threshold:
                    catastrophic_count += right - left
                    right -= 1
                else:
                    left += 1

        return catastrophic_count / total_combinations



def random_cluster_catastrophic_probability(
    values: List[float],
    k: int,
    threshold: float = CATASTROPHIC_THRESHOLD,
) -> float:
    
    """
    this func calculates exact probability that a uniformly random contiguous k-token deletion
    loses at least threshold of the sentence information
    """
    total = sum(values)
    n = len(values)
    k = 3
    if total <= 0 or n == 0 or k <= 0:
        return 0.0
    if k >= n:
        return 1.0

    number_of_windows = n - k + 1           # we have k = 3, n = 10, so number_of_windows = 8
    catastrophic_count = 0
    window = sum(values[:k])            # first 3 block

    if window / total >= threshold:
        catastrophic_count += 1

    for i in range(k, n):           # again sliding window, instead of recalculating 3 values each time
        window += values[i] - values[i-k]      # add new token entering the window and subtract the old token leaving the window
        if window / total >= threshold:
            catastrophic_count += 1

    return catastrophic_count / number_of_windows


def init_bucket() -> Dict[str, Any]:
    return {
        "rows": 0,
        "sum_total_bits": 0.0,
        "sum_token_count": 0.0,
        "sum_avg_token_surprisal": 0.0,
        "sum_max_token_share": 0.0,     # accumulates largest token IC / sentence total IC
        "count_max_token_ge_50": 0,     # Counts sentences where a single token itself contains ≥50% information
        "sum_top_k_share": 0.0,         # Stores total of the information share contained in the top 3 highest surprisal tokens
        "count_top_k_ge_50": 0,     # Counts how many sentences have their 3 most informative words containing ≥50% total information, tells us whether a catastrophic 3-word deletion is possible
        "sum_max_cluster_share": 0.0,       # Stores maximum possible information share in any contiguous 3-word block
        "count_any_cluster_ge_50": 0,       # Counts sentences containing at least one contiguous 3-word block with ≥50% information, to know that catastrophic clustered deletion is possible somewhere
        "sum_first_k_share": 0.0,       # information concentrated specifically in the beginning of the sentence  - asymmetric analysis
        "count_first_k_ge_50": 0,   # no. of sentences whose first 3 words already contains ≥50% information
        "sum_information_center": 0.0,      # Accumulates information centre
        "sum_single_theoretical_catastrophic_probability": 0.0,
        "sum_cluster_theoretical_catastrophic_probability": 0.0,
        "single_noise_rows": 0,
        "sum_single_proportion_bits_lost": 0.0,
        "count_single_catastrophic_loss": 0,
        "clustered_noise_rows": 0,
        "sum_clustered_proportion_bits_lost": 0.0,
        "count_clustered_catastrophic_loss": 0,
        "highest_order_context_diag_tokens": 0,             # token level diagnostic  
        "highest_order_context_seen": 0,            
        "highest_order_ngram_diag_tokens": 0,
        "highest_order_ngram_seen": 0,
        "scoring_level_ngram_diag_tokens": 0,
        "scoring_level_ngram_seen": 0,
        "scoring_level_tokens": 0,
        "scoring_trigram_tokens": 0,
        "scoring_bigram_tokens": 0,
        "scoring_unigram_tokens": 0,
        "scoring_floor_tokens": 0,
        "floor_diag_tokens": 0,
        "floor_tokens": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_summary_csv", required=False)
    args = parser.parse_args()

    path = Path(args.input_csv)
    if not path.exists():
        raise FileNotFoundError(path)

    stats = defaultdict(init_bucket)    # dict that  accumulates statistics across all chunks 
    rows_seen = 0                       # Number of input rows processed
    next_progress = PROGRESS_EVERY

    single_noise_columns = {
        "single_proportion_bits_lost",
        "single_catastrophic_loss",
    }
    clustered_noise_columns = {
        "clustered_proportion_bits_lost",
        "clustered_catastrophic_loss",
    }

    usecols = ["type", "token_ic_json", "total_bits", "single_proportion_bits_lost", "single_catastrophic_loss",
               "clustered_proportion_bits_lost","clustered_catastrophic_loss"]

    reader = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=usecols, chunksize=CHUNK_SIZE)      # reader not a dataframe, it is an iterator
   
    for chunk in reader:        # each time we  request the next element, pandas loads another 50,000 rows
        for row in chunk.itertuples(index=False):        # itertuples is faster than iterrows
            version = str(row.type).strip()                     # sentence version - actual, asymmetric, random, uniform 
            records = parse_records(row.token_ic_json)
            values = [float(r["information_content"]) for r in records]
            n = len(values)
            if n == 0:
                continue
            
            total = float(sum(values))

            k = deletion_count(n)           # tried to make generic, but actually k = 3 only in our experiment
            shares = [v / total for v in values] if total > 0 else [0.0] * n    # convert token IC into information shares    
            max_token_share = max(shares) if shares else 0.0        # largest individual token share
            top_k_share = sum(sorted(shares, reverse=True)[:k])     # Top 30% information concentration - information contained in 3 most informative words
            max_cluster_share = max_contiguous_share(values, k) # finds the most information-rich consecutive 3 words - for clustered noise analysis
            first_k_share = sum(values[:k]) / total if total > 0 else 0.0       # information at sentence beginning - first 3 token share
            center = information_center(values)
            single_theoretical_probability = random_k_catastrophic_probability(shares, k)  # considers every possible k-position deletion combination
            cluster_theoretical_probability = random_cluster_catastrophic_probability(values,k) # theoretical cluster probability
            
            if single_theoretical_probability is None:
                print(
                    f"Problem: version={version}, "
                    f"sentence_length={n}, k={k}"
                )
                raise ValueError("Unsupported deletion count")


            b = stats[version]          # eg. b = stats["Actual"]
            b["rows"] += 1
            b["sum_total_bits"] += total
            b["sum_token_count"] += n
            b["sum_avg_token_surprisal"] += total / n
            b["sum_max_token_share"] += max_token_share             # accumulate maximum-token concentration
            b["count_max_token_ge_50"] += int(max_token_share >= CATASTROPHIC_THRESHOLD)        # int(True) = 1, so counter like
            b["sum_top_k_share"] += top_k_share
            b["count_top_k_ge_50"] += int(top_k_share >= CATASTROPHIC_THRESHOLD)
            b["sum_max_cluster_share"] += max_cluster_share
            b["count_any_cluster_ge_50"] += int(max_cluster_share >= CATASTROPHIC_THRESHOLD)
            b["sum_first_k_share"] += first_k_share
            b["count_first_k_ge_50"] += int(first_k_share >= CATASTROPHIC_THRESHOLD)
            b["sum_information_center"] += center
            b["sum_single_theoretical_catastrophic_probability"] += single_theoretical_probability
            b["sum_cluster_theoretical_catastrophic_probability"] += cluster_theoretical_probability
            b["single_noise_rows"] += 1
            b["sum_single_proportion_bits_lost"] += float(row.single_proportion_bits_lost)
            b["count_single_catastrophic_loss"] += int(float(row.single_catastrophic_loss) >= 0.5)
            b["clustered_noise_rows"] += 1
            b["sum_clustered_proportion_bits_lost"] += float(row.clustered_proportion_bits_lost)
            b["count_clustered_catastrophic_loss"] += int(float(row.clustered_catastrophic_loss) >= 0.5)

            # LM diagnostics stored in token_ic_json

            for rec in records:
                if "highest_order_context_seen" in rec:
                    b["highest_order_context_diag_tokens"] += 1
                    if bool(rec["highest_order_context_seen"]):
                        b["highest_order_context_seen"] += 1

                if "highest_order_ngram_seen" in rec:
                    b["highest_order_ngram_diag_tokens"] += 1
                    if bool(rec["highest_order_ngram_seen"]):
                        b["highest_order_ngram_seen"] += 1

                if "scoring_level_ngram_seen" in rec:
                    b["scoring_level_ngram_diag_tokens"] += 1
                    if bool(rec["scoring_level_ngram_seen"]):
                        b["scoring_level_ngram_seen"] += 1

                if "scoring_level" in rec:
                    scoring_level = str(rec["scoring_level"]).strip().lower()
                    b["scoring_level_tokens"] += 1

                    if scoring_level == "trigram":
                        b["scoring_trigram_tokens"] += 1
                    elif scoring_level == "bigram":
                        b["scoring_bigram_tokens"] += 1
                    elif scoring_level == "unigram":
                        b["scoring_unigram_tokens"] += 1
                    elif scoring_level == "floor":
                        b["scoring_floor_tokens"] += 1

                if "used_floor_probability" in rec:
                    b["floor_diag_tokens"] += 1
                    if bool(rec["used_floor_probability"]):
                        b["floor_tokens"] += 1

        rows_seen += len(chunk)
        if rows_seen >= next_progress:      # we check whether progress reporting threshold has been crossed
            print(f"Processed {rows_seen:,} rows", flush=True)
            while next_progress <= rows_seen:
                next_progress += PROGRESS_EVERY

    rows = []
    #versions = VERSION_ORDER + sorted(v for v in stats if v not in VERSION_ORDER)
    versions = VERSION_ORDER
    for version in versions:
        if version not in stats or stats[version]["rows"] == 0:
            continue
        b = stats[version]
        nrows = b["rows"]
        single_rows = b["single_noise_rows"]
        clustered_rows = b["clustered_noise_rows"]
        context_dtok = b["highest_order_context_diag_tokens"]
        highest_ngram_dtok = b["highest_order_ngram_diag_tokens"]
        scoring_ngram_dtok = b["scoring_level_ngram_diag_tokens"]
        scoring_dtok = b["scoring_level_tokens"]
        floor_dtok = b["floor_diag_tokens"]

        rows.append({
            "type": version,
            "rows": nrows,
            "mean_total_bits": b["sum_total_bits"] / nrows,
            "mean_token_count": b["sum_token_count"] / nrows,
            "mean_avg_token_surprisal": b["sum_avg_token_surprisal"] / nrows,
            "mean_max_token_share": b["sum_max_token_share"] / nrows,
            "pct_sentences_one_token_ge_50": 100.0 * b["count_max_token_ge_50"] / nrows,
            "mean_top_30pct_token_share": b["sum_top_k_share"] / nrows,
            "pct_sentences_top_30pct_ge_50": 100.0 * b["count_top_k_ge_50"] / nrows,
            "pct_single_theoretical_catastrophic_loss": 100.0 * b["sum_single_theoretical_catastrophic_probability"] / nrows,
            "mean_max_contiguous_30pct_share": b["sum_max_cluster_share"] / nrows,
            "pct_sentences_any_cluster_ge_50": 100.0 * b["count_any_cluster_ge_50"] / nrows,
            "pct_cluster_theoretical_catastrophic_loss": 100.0 * b["sum_cluster_theoretical_catastrophic_probability"] / nrows,
            "mean_first_30pct_share": b["sum_first_k_share"] / nrows,
            "pct_sentences_first_30pct_ge_50": 100.0 * b["count_first_k_ge_50"] / nrows,
            "mean_information_center": b["sum_information_center"] / nrows,
            "mean_single_proportion_bits_lost": (b["sum_single_proportion_bits_lost"] / single_rows) if single_rows else float("nan"),
            "pct_single_catastrophic_loss": (100.0 * b["count_single_catastrophic_loss"] / single_rows) if single_rows else float("nan"),
            "mean_clustered_proportion_bits_lost": (b["sum_clustered_proportion_bits_lost"] / clustered_rows) if clustered_rows else float("nan"),
            "pct_clustered_catastrophic_loss": (100.0 * b["count_clustered_catastrophic_loss"] / clustered_rows) if clustered_rows else float("nan"),
            "pct_tokens_highest_order_context_seen": (100.0 * b["highest_order_context_seen"] / context_dtok) if context_dtok else float("nan"),
            "pct_tokens_highest_order_ngram_seen": (100.0 * b["highest_order_ngram_seen"] / highest_ngram_dtok) if highest_ngram_dtok else float("nan"),
            "pct_tokens_scoring_level_ngram_seen": (100.0 * b["scoring_level_ngram_seen"] / scoring_ngram_dtok) if scoring_ngram_dtok else float("nan"),
            "mean_tokens_scoring_trigram": b["scoring_trigram_tokens"] / nrows,
            "mean_tokens_scoring_bigram": b["scoring_bigram_tokens"] / nrows,
            "mean_tokens_scoring_unigram": b["scoring_unigram_tokens"] / nrows,
            "mean_tokens_scoring_floor": b["scoring_floor_tokens"] / nrows,
            "pct_tokens_scoring_trigram": (100.0 * b["scoring_trigram_tokens"] / scoring_dtok) if scoring_dtok else float("nan"),
            "pct_tokens_scoring_bigram": (100.0 * b["scoring_bigram_tokens"] / scoring_dtok) if scoring_dtok else float("nan"),
            "pct_tokens_scoring_unigram": (100.0 * b["scoring_unigram_tokens"] / scoring_dtok) if scoring_dtok else float("nan"),
            "pct_tokens_scoring_floor": (100.0 * b["scoring_floor_tokens"] / scoring_dtok) if scoring_dtok else float("nan"),
            "pct_tokens_floor_probability": (100.0 * b["floor_tokens"] / floor_dtok) if floor_dtok else float("nan"),
        })

    summary = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    concentration_columns = [
        "type",
        "rows",
        "mean_total_bits",
        "mean_token_count",
        "mean_avg_token_surprisal",
        "mean_max_token_share",
        "pct_sentences_one_token_ge_50",
        "mean_top_30pct_token_share",
        "pct_sentences_top_30pct_ge_50",
        "mean_max_contiguous_30pct_share",
        "pct_sentences_any_cluster_ge_50",
        "mean_first_30pct_share",
        "pct_sentences_first_30pct_ge_50",
        "mean_information_center",
    ]

    noise_columns = [
        "type",
        "pct_sentences_top_30pct_ge_50",
        "pct_single_theoretical_catastrophic_loss",
        "mean_single_proportion_bits_lost",
        "pct_single_catastrophic_loss",
        "pct_sentences_any_cluster_ge_50",
        "pct_cluster_theoretical_catastrophic_loss",
        "mean_clustered_proportion_bits_lost",
        "pct_clustered_catastrophic_loss",
    ]

    lm_columns = [
        "type",
        "pct_tokens_highest_order_context_seen",
        "pct_tokens_highest_order_ngram_seen",
        "pct_tokens_scoring_level_ngram_seen",
        "mean_tokens_scoring_trigram",
        "mean_tokens_scoring_bigram",
        "mean_tokens_scoring_unigram",
        "mean_tokens_scoring_floor",
        "pct_tokens_scoring_trigram",
        "pct_tokens_scoring_bigram",
        "pct_tokens_scoring_unigram",
        "pct_tokens_scoring_floor",
        "pct_tokens_floor_probability",
    ]

    print("\n========== Contextual Surprisal Concentration Diagnostics ==========")
    print(summary[concentration_columns].to_string(index=False))

    print("\n========== Random Noise / Catastrophic-Loss Diagnostics ==========")
    print(summary[noise_columns].to_string(index=False))

    print("\n========== Language-Model Scoring Diagnostics ==========")
    print(summary[lm_columns].to_string(index=False))

    if args.output_summary_csv:
        out = Path(args.output_summary_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nSummary written to: {out}")


if __name__ == "__main__":
    main()
