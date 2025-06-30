"""Convert results data to a LaTeX table."""

from pathlib import Path

import pandas as pd

RESULTS_DIRECTORY: Path = Path("./assets/results/")
TYPES = {
    "vanilla": "None",
    "jacobi": "Jacobi",
    "incomplete_cholesky": "\\ac{ic}(0)",
    "incomplete_lu": "\\ac{ilu}(0)",
    "learned": "Ours",
}
HEADER = {
    "kappas": "$\\kappa$",
    "densities": "Density [\\si{\\percent}]",
    "iterations": "Iters",
    "setups": "Setup",
    "durations": "Converge",
    "totals": "Total [\\si{\\second}]",
}


def summary_stats_to_latex() -> str:
    """Convert results data to a LaTeX table.

    Returns:
        The LaTeX code for the table.
    """
    combined = pd.DataFrame()

    for parameter in ["kappas", "densities", "iterations", "setups", "durations", "totals"]:
        data = pd.read_csv(RESULTS_DIRECTORY / f"all_{parameter}.csv")

        stats = []
        for technique in data.select_dtypes(include="number").columns:
            mean_val = data[technique].mean()
            std_val = data[technique].std()
            if parameter in ["kappas", "iterations"]:
                stats.append(f"${mean_val:.0f} \\pm {std_val:.0f}$")
            else:
                stats.append(f"${mean_val:.2f} \\pm {std_val:.2f}$")
        label = HEADER[parameter]
        temp_df = pd.DataFrame({label: stats}, index=data.select_dtypes(include="number").columns)
        combined = pd.concat([combined, temp_df], axis=1)

    combined.index = combined.index.to_series().replace(TYPES)
    combined.index.name = "Technique"
    combined.reset_index(inplace=True)

    return combined.to_latex(index=False, escape=False, column_format="r" * len(combined.columns))


def main() -> None:
    """Main function to generate the LaTeX table from results data."""
    table_latex = summary_stats_to_latex()
    with Path("/tmp/table.tex").open(mode="w") as file_io:
        file_io.write(table_latex)


if __name__ == "__main__":
    main()
