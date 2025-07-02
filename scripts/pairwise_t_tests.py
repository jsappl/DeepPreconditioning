"""Compute pairwise t-tests for all combinations of columns in a CSV file."""

import itertools
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

THRESHOLD = 0.05

# Load your CSV file
total_times = pd.read_csv(Path("./assets/results/all_totals.csv"))

# Prepare to store results
results = []

# Generate all unique pairs of columns
for column in total_times.columns:
    if column == "learned":
        continue

    # Drop rows where either value is NaN
    paired_data = total_times[["learned", column]].dropna()
    # Perform paired t-test
    t_stat, p_val = ttest_rel(paired_data["learned"], paired_data[column])
    results.append(
        dict(
            technique_1="learned",
            technique_2=column,
            t_statistic=t_stat,
            p_value=p_val,
            significant=p_val < THRESHOLD,
        )
    )

# Convert results to DataFrame and print
print(pd.DataFrame(results))
