
import csv
from pathlib import Path


INPUT_FILE = Path("./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv")

OUTPUT_FILE = Path("./outputs/mtp_2_features/exp_4_demo_output/stats_ip_file_demo.csv")

columns_to_remove = {
    "token_ic_json",
    "single_deleted_positions_json",
    "single_deleted_count",
    "single_corrupted_sentence",
    "clustered_deleted_positions_json",
    "clustered_deleted_count",
    "clustered_corrupted_sentence",
}

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as outfile:

    infile.seek(0)

    reader = csv.DictReader(infile, delimiter=",")

    keep_columns = [
        col for col in reader.fieldnames
        if col not in columns_to_remove
    ]

    writer = csv.DictWriter(outfile, fieldnames=keep_columns)
    writer.writeheader()

    for row in reader:
        writer.writerow({col: row[col] for col in keep_columns})

print("Cleaned file saved:", OUTPUT_FILE)
print("Remaining columns:", keep_columns)
