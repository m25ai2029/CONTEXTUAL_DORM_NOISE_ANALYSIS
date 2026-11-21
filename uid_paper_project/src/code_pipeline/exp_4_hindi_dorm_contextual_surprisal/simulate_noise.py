#!/usr/bin/env python3
"""
himanshu- version 2 - chunks processing added

This file simulates single-word random-position noise and clustered noise
for our 4 sentence versions, then compute proportional bit loss

The input CSV is processed in chunks so that very large files do not need to
be loaded into memory at once.

Input CSV columns:
    sent_count, type, sentence, DORM, token_ic_json, total_bits

Outputs:
    One  CSV file containing the original sentence data plus
       single-noise and clustered-noise results.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, TextIO, Tuple

import pandas as pd


DEFAULT_RANDOM_SEED = 42
DEFAULT_NOISE_FRACTION = 0.30
VALID_NOISE_TYPES = ("single", "clustered")

# Number of rows read from disk at one time.
# This is deliberately smaller than PROGRESS_EVERY to keep memory bounded.
READ_CHUNK_SIZE = 50_000

# Print a progress message whenever another 100,000 input rows have been read.
PROGRESS_EVERY = 100_000

def tokenize_sentence(sentence: str) -> List[str]:

    """ whitespace tokenizer matching the generated sentence column."""
    return str(sentence).strip().split()


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_token_ic_json(json_text: str) -> List[float]:

    """ thisw func parse token_ic_json field and return IC values from that row in a float list."""
    if pd.isna(json_text):
        raise ValueError("token_ic_json is missing.")

    parsed = json.loads(str(json_text))

    if not isinstance(parsed, list):
        raise ValueError("token_ic_json must be a JSON list.")

    ic_values: List[float] = []

    for record in parsed:
        ic_values.append(float(record["information_content"]))

    return ic_values


def validate_noise_type(noise_type: str) -> None:
    """Raise an error if an unsupported noise type is supplied."""
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(
            f"noise_type must be one of {VALID_NOISE_TYPES}, got {noise_type!r}."
        )


# ---------------------------------------------------------------------
# Noise simulation
# ---------------------------------------------------------------------

def deletion_count(sentence_length: int, noise_fraction: float) -> int:
    """
    this function computes number of tokens to delete

    For length 10 and fraction 0.30, this gives 3.
    """
    if sentence_length <= 0:
        return 0

    count = int(round(sentence_length * noise_fraction))
    count = max(1, count)
    count = min(sentence_length, count)
    return count


def choose_single_noise_positions(
    sentence_length: int,
    noise_fraction: float,
    rng: random.Random,
) -> List[int]:
    
    """ this function chooses random positions-  they do not need to be consecutive """

    count = deletion_count(sentence_length, noise_fraction)  # 3, made a bit generic
    positions = rng.sample(range(sentence_length), k=count)     # rng.sample(range(10), k=3) -> eg [7,3,5]
    positions.sort()  # [3,5,7]   sorts it
    return positions


def choose_clustered_noise_positions(
    sentence_length: int,
    noise_fraction: float,
    rng: random.Random,
) -> List[int]:
    """Choose one consecutive block of positions."""
    count = deletion_count(sentence_length, noise_fraction)

    if count >= sentence_length:
        return list(range(sentence_length))

    start = rng.randint(0, sentence_length - count)
    return list(range(start, start + count))            # eg, range(4,4+3)  gives [4,5,6]


def delete_positions_from_tokens(
    tokens: Sequence[str],
    positions: Sequence[int],
) -> str:
    
    """ this function returns remaining sentence after deleting selected token positions."""
    position_set = set(positions)
    kept_tokens = []

    for index, token in enumerate(tokens):
        if index not in position_set:
            kept_tokens.append(token)

    return " ".join(kept_tokens)


def sum_deleted_bits(
    ic_values: Sequence[float],
    positions: Sequence[int],
) -> float:
    
    """ this function will sum IC values at deleted positions."""
    total = 0.0

    for position in positions:
        if 0 <= position < len(ic_values):
            total += float(ic_values[position])

    return total


def prepare_dataframe_chunk(df: pd.DataFrame) -> pd.DataFrame:
   
    """ this fnc cleans and converts one input chunk."""
    # Remove accidental spaces from column names.
    df.columns = [column.strip() for column in df.columns]

    required_columns = {
        "sent_count",
        "type",
        "sentence",
        "DORM",
        "token_ic_json",
        "total_bits",
    }
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Remove accidental leading/trailing spaces.
    df["sent_count"] = df["sent_count"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip()
    df["sentence"] = df["sentence"].fillna("").astype(str).str.strip()

    # Convert numeric columns from strings to numeric values.
    df["DORM"] = pd.to_numeric(df["DORM"])
    df["total_bits"] = pd.to_numeric(df["total_bits"])

    return df


def split_complete_rows_and_pending_group(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A sent_count group may be cut in half when pandas reads the CSV in chunks.
    this func holds back the final sent_count group from a chunk.

    The held-back rows are prepended to the next input chunk. This prevents a
    sentence group split across two chunks from receiving two independently
    generated noise-position sets.

    """
    if df.empty:
        return df, df.copy()

    
    # last_sent_count  - last row sent_count
    # pending_mask - True for all last rows whose sent_count matches with last_sent_count

    last_sent_count = str(df.iloc[-1]["sent_count"])   # Get the last row, last sent-count in that chunk (50000 rows)
    
    pending_mask = df["sent_count"].astype(str) == last_sent_count      # create mask, [F, F, F, F, T]

    process_df = df.loc[~pending_mask].copy()       # groups that are safe to process now
    pending_df = df.loc[pending_mask].copy()        # final group, saved for the next chunk

    return process_df, pending_df


def build_group_noise_position_map(
    df: pd.DataFrame,                   # current complete chunk of dataframe/ complete groups being processed
    noise_fraction: float,
    rng: random.Random,
) -> Dict[str, Dict[str, List[int]]]:
    
    """
    the function chooses noise once per sentence group, rather than separately for every row, i.e. 
    this function chooses one single noise position set and one clustered noise position set
    for each sent_count group.
    The same selected positions are applied to all four versions of that
    sentence group, matching with the the paper.
    returns something like:
    position_map = {

    "101": {
        "single": [1, 5, 7],
        "clustered": [4, 5, 6]
            },

    "102": {
        "single": [0, 3, 8],
        "clustered": [6, 7, 8]
            },

    "103": {
        "single": [2, 4, 9],
        "clustered": [0, 1, 2]
            }
    }

    """
    position_map: Dict[str, Dict[str, List[int]]] = {}

    """
    sent_count = 101
    group_df =
    Actual
    Uniform
    Random
    Asymmetric
    """


    for sent_count, group_df in df.groupby("sent_count", sort=False):
        first_row = group_df.iloc[0]            # fetches first row from group_df, group of 4 variants, i.e. selects "Actual"
        ic_values = parse_token_ic_json(first_row["token_ic_json"])  # fetches all IC values in a list
        sentence_length = len(ic_values)            # calculates length of each IC vector, i.e. sentence length

        single_noise_positions = choose_single_noise_positions(sentence_length, noise_fraction, rng)
        clustered_noise_positions = choose_clustered_noise_positions(sentence_length, noise_fraction, rng)
        
        position_map[str(sent_count)] = {
            "single": single_noise_positions,
            "clustered": clustered_noise_positions,
        }

    return position_map


def add_noise_results(df: pd.DataFrame, noise_fraction: float, rng: random.Random) -> pd.DataFrame:
    """
    this func adds single noise and clustered noise results to one dataframe chunk
    One output row is produced for each input row. 
    """
    if df.empty:
        return df.copy()

    group_position_map = build_group_noise_position_map(df,noise_fraction,rng)

    wide_rows: List[Dict[str, Any]] = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()

        sent_count = str(row_dict["sent_count"])
        sentence = row_dict["sentence"]

        tokens = tokenize_sentence(sentence)
        ic_values = parse_token_ic_json(row_dict["token_ic_json"])
        total_bits = safe_float(row_dict["total_bits"])

        # Copy all original columns into the output row.
        output_row = dict(row_dict)

        for noise_type in VALID_NOISE_TYPES:
            positions = group_position_map[sent_count][noise_type]   # select position map for that particular sent count
            deleted_bits = sum_deleted_bits(ic_values, positions)

            proportion_bits_lost = 0.0
            if total_bits > 0:
                proportion_bits_lost = deleted_bits / total_bits

            catastrophic_loss = 1 if proportion_bits_lost >= 0.50 else 0
            corrupted_sentence = delete_positions_from_tokens(tokens, positions)

            output_row[f"{noise_type}_deleted_positions_json"] = json.dumps(positions,ensure_ascii=False,)
            output_row[f"{noise_type}_deleted_count"] = len(positions)
            output_row[f"{noise_type}_deleted_bits"] = deleted_bits
            output_row[f"{noise_type}_proportion_bits_lost"] = proportion_bits_lost
            output_row[f"{noise_type}_catastrophic_loss"] = catastrophic_loss
            output_row[f"{noise_type}_corrupted_sentence"] = corrupted_sentence

        wide_rows.append(output_row)

    return pd.DataFrame(wide_rows)


def write_output_chunk(
    wide_df: pd.DataFrame,
    output_handle: TextIO,
    write_header: bool,
) -> None:
    
    """this function appends one processed wide chunk to the already-open output CSV."""

    wide_df.to_csv(output_handle, index=False, header=write_header)


def print_progress(
    rows_read: int,
    rows_processed: int,
    rows_written: int,
    sentence_groups_processed: int,
    pending_rows: int,
    start_time: float,
) -> None:
    
    """ this function prints cumulative chunk-processing progress """
    elapsed_seconds = max(time.perf_counter() - start_time, 1e-9)
    processing_rate = rows_processed / elapsed_seconds

    print(
        "Progress: "
        f"read={rows_read:,} rows | "
        f"processed={rows_processed:,} | "
        f"written={rows_written:,} | "
        f"groups={sentence_groups_processed:,} | "
        f"pending={pending_rows:,} | "
        f"rate={processing_rate:,.0f} processed rows/s",
        flush=True,
    )


def process_csv_in_chunks(
    input_csv: str,
    output_csv: str,
    noise_fraction: float,
    random_seed: int,
    chunk_size: int,
    progress_every: int,
) -> Tuple[int, int, int]:
    
    """
    this function read, process and write the large CSV incrementally.

    Returns:
        rows_read: Number of source rows read.
        rows_written: Number of processed rows written to the output CSV.
        sentence_groups_processed: Number of sent_count groups processed.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    
    # .stat() asks the operating system for information/metadata about the file at input_path
    # st_size - size of the file in bytes

    input_size_gb = input_path.stat().st_size / (1024 ** 3)

    print(" -  Chunked Noise Simulation  - ")
    print(f"Input file          : {input_path}")
    print(f"Input file size     : {input_size_gb:.2f} GB")
    print(f"Output file         : {output_path}")
    print(f"Read chunk size     : {chunk_size:,} rows")
    print(f"Progress every      : {progress_every:,} rows")
    print(f"Noise fraction      : {noise_fraction:.2f}")
    print(f"Random seed         : {random_seed}")
    print("Starting processing...", flush=True)

    # perf-counter  - used to record the current high-precision timer value
    # returns a high-resolution performance counter in seconds

    start_time = time.perf_counter()
    rng = random.Random(random_seed)

    rows_read = 0
    rows_processed = 0
    rows_written = 0
    sentence_groups_processed = 0
    next_progress = progress_every
    header_written = False

    pending_df = pd.DataFrame()

    '''
    # read csv in chunks
    pd.read_csv() does not return one full dataframe because chunksize is specified, instead, 
    it returns a TextFileReader object, we can then iterate it over, eg:
    for chunk in reader:
        print(chunk)
    here each chunk is a normal pandas dataframe containing chunk_size rows   
    '''


    reader = pd.read_csv(input_path, encoding="utf-8", dtype=str, chunksize=chunk_size)

    # in below code- we keep the output file open throughout the entire run.

    with output_path.open("w", encoding="utf-8-sig", newline="") as output_handle:

        for raw_chunk in reader:
            rows_read += len(raw_chunk)
            chunk_df = prepare_dataframe_chunk(raw_chunk)

            if not pending_df.empty:
                chunk_df = pd.concat([pending_df, chunk_df], ignore_index=True)
                pending_df = pd.DataFrame()

            process_df, pending_df = split_complete_rows_and_pending_group(chunk_df)

            if not process_df.empty:
                wide_chunk = add_noise_results(df=process_df, noise_fraction=noise_fraction,rng=rng)

                write_output_chunk(wide_df=wide_chunk, output_handle=output_handle, write_header=not header_written)
                header_written = True

                rows_processed += len(process_df)
                rows_written += len(wide_chunk)
                sentence_groups_processed += process_df["sent_count"].nunique()

                # we explicitly release the large chunk objects as soon as their
                # output has been written.
                del wide_chunk
                del process_df

            del chunk_df
            del raw_chunk

            if rows_read >= next_progress:
                print_progress(
                    rows_read=rows_read,
                    rows_processed=rows_processed,
                    rows_written=rows_written,
                    sentence_groups_processed=sentence_groups_processed,
                    pending_rows=len(pending_df),
                    start_time=start_time,
                )

                while next_progress <= rows_read:
                    next_progress += progress_every

        # The final held-back sent_count group is now known to be complete
        # because the input file has ended, so process it once.
        if not pending_df.empty:
            wide_chunk = add_noise_results(df=pending_df, noise_fraction=noise_fraction, rng=rng)

            write_output_chunk(wide_df=wide_chunk, output_handle=output_handle, write_header=not header_written)
            header_written = True

            rows_processed += len(pending_df)
            rows_written += len(wide_chunk)
            sentence_groups_processed += pending_df["sent_count"].nunique()

            pending_df = pd.DataFrame()
            del wide_chunk

    if rows_read == 0:
        raise ValueError("Input CSV contains no data rows.")

    if rows_written != rows_read:
        raise RuntimeError(
            "Row-count mismatch after chunk processing: "
            f"read {rows_read:,}, wrote {rows_written:,}."
        )

    print_progress(
        rows_read=rows_read,
        rows_processed=rows_processed,
        rows_written=rows_written,
        sentence_groups_processed=sentence_groups_processed,
        pending_rows=0,
        start_time=start_time,
    )

    print("Chunk processing finished successfully.", flush=True)

    return (
        rows_read,
        rows_written,
        sentence_groups_processed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate single random position and clustered  noise "
             "compute proportional bit loss "
        )
    )

    parser.add_argument("--input_csv", required=True, help="Input CSV file.")
    parser.add_argument("--output_csv",required=True,help="Output wide CSV file path.")

    args = parser.parse_args()

    (
        total_input_rows,
        total_output_rows,
        sentence_groups,
    ) = process_csv_in_chunks(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        noise_fraction=DEFAULT_NOISE_FRACTION,
        random_seed=DEFAULT_RANDOM_SEED,
        chunk_size=READ_CHUNK_SIZE,
        progress_every=PROGRESS_EVERY,
    )

    print("\nNoise simulation output has been written.")

    print("\nFiles written:")
    print(f"Output CSV file                 : {args.output_csv}")
    print(f"Input rows                      : {total_input_rows:,}")
    print(f"Output rows                     : {total_output_rows:,}")
    print(f"Sentence groups                 : {sentence_groups:,}")


if __name__ == "__main__":
    main()
