
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from pathlib import Path

INPUT_FILE = Path("./outputs/mtp_2_features/exp_4_demo_output/stats_ip_file_demo.csv")
OUTPUT_FILE = Path("./outputs/mtp_2_features/exp_4_demo_output/statistical_results.csv")

# read only the required columns
df = pd.read_csv(INPUT_FILE,  usecols=["sent_count", "type", "DORM"])

# convert DORM to numeric
df["DORM"] = pd.to_numeric(df["DORM"], errors="coerce")

# Each row = one original sentence
# Each column = one sentence version and DORM values populated below each column

wide_df = df.pivot(index="sent_count", columns="type", values="DORM")

# list of column names representing all version types to evaluate
versions = ["Actual", "Uniform", "Asymmetric", "Random"]
wide_df = wide_df.dropna(subset=versions)       # Drop rows that contain missing values in any of the specified version columns

# Comparisons to be done- 
comparisons = [
    ("Uniform", "Actual"),
    ("Uniform", "Random"),
    ("Uniform", "Asymmetric"),
    ("Actual", "Random"),
    ("Actual", "Asymmetric"),
]

results = []

# iterate through each pair of versions to compare
for version1, version2 in comparisons:

    x = wide_df[version1].to_numpy()            # extract data as numpy arrays, x = DORM of first version
    y = wide_df[version2].to_numpy()            # y = dorm of second  variant against which to compare 

    # Paired t-test
    t_stat, p_value = ttest_rel(x, y)

    # Difference for each original sentence
    differences = x - y

    # Cohen's d for paired observations (d_z)

    # Compute sample standard deviation of differences (ddof=1 for sample estimation)
    
    std_diff = np.std(differences, ddof=1)   
    # calculate Cohen's d_z effect size for paired observations (mean diff / std diff)           
    cohen_d = (
        np.mean(differences) / std_diff
        if std_diff > 0 else np.nan
    )

    # stores calculated metrics and test statistics

    results.append({
        "Comparison": f"{version1} vs {version2}",
        "N": len(differences),
        #"Mean_1": np.mean(x),
        #"Mean_2": np.mean(y),
        "Mean_Difference": np.mean(differences),
        "t_statistic": t_stat,
        "df": len(differences) - 1,
        "p_value": p_value,
        #"Cohens_d_signed": cohen_d,
        "Cohens_d_absolute": abs(cohen_d)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print(results_df.to_string(index=False))
print("Results saved to:", OUTPUT_FILE)
