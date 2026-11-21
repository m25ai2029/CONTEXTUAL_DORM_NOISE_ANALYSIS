import csv
from pathlib import Path
import argparse

def extract_top_n_rows(input_csv: str, output_csv: str, n_rows: int = 100) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as infile, \
         output_path.open("w", encoding="utf-8", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header
        header = next(reader, None)
        if header is None:
            print("Input CSV is empty.")
            return

        writer.writerow(header)

        # Write up to n_rows data rows
        count = 0
        for row in reader:
            writer.writerow(row)
            count += 1
            if count >= n_rows:
                break

    print(f"Done. Wrote {count} rows to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="write sample rows for preview of file")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    args = parser.parse_args()
    n_rows = 500
    extract_top_n_rows(input_csv = args.input_csv, output_csv = args.output_csv, n_rows = n_rows)