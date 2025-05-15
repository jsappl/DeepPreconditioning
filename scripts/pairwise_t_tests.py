"""Compute pairwise t-tests for all combinations of columns in a CSV file."""

import itertools
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

THRESHOLD = 0.05

# Load your CSV file
total_times = pd.read_csv(Path("./assets/results/totals.csv"))

# Prepare to store results
results = []

# Generate all unique pairs of columns
for col1, col2 in itertools.combinations(total_times.columns, 2):
    # Drop rows where either value is NaN
    paired_data = total_times[[col1, col2]].dropna()
    # Perform paired t-test
    t_stat, p_val = ttest_rel(paired_data[col1], paired_data[col2])
    results.append(
        dict(
            technique_1=col1,
            technique_2=col2,
            t_statistic=t_stat,
            p_value=p_val,
            significant=p_val < THRESHOLD,
        )
    )

# Convert results to DataFrame and print
print(pd.DataFrame(results))
