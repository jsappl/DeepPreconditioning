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
    "setups": "Setup [\\si{\\second}]",
    "durations": "Converge [\\si{\\second}]",
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


def compute_boxplot_stats(df):
    """Compute boxplot statistics."""
    stats = {}

    for col in df.columns:
        data = df[col].dropna().sort_values()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        median = data.median()
        iqr = q3 - q1
        lower_whisker = data[data >= (q1 - 1.5 * iqr)].min()
        upper_whisker = data[data <= (q3 + 1.5 * iqr)].max()
        outliers = data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))].tolist()

        stats[TYPES[col]] = {
            "q1": q1,
            "median": median,
            "q3": q3,
            "lower_whisker": lower_whisker,
            "upper_whisker": upper_whisker,
            "outliers": outliers,
        }

    return stats


def tikz_from_boxplot_stats(stats, ylabel: str, filepath="/tmp/boxplot.tex", outliers: bool = False):
    """Generate TikZ/PGFPlots LaTeX code for manual boxplots from precomputed stats."""
    tikz = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "    ymode=log,",
        "    boxplot/draw direction=y,",
        f"    xtick={{1,...,{len(stats)}}},",
        "    xticklabels={%s}," % ",".join(stats.keys()),
        "    x tick label style={rotate=45, anchor=east},",
        "    xlabel={Technique},",
        f"    ylabel={{{ylabel}}},",
        "    width=7cm,",
        " ]",
    ]

    for idx, (_, s) in enumerate(stats.items(), start=1):
        tikz += [
            "\\addplot+ [",
            "    boxplot prepared={",
            f"        median={s['median']},",
            f"        upper quartile={s['q3']},",
            f"        lower quartile={s['q1']},",
            f"        upper whisker={s['upper_whisker']},",
            f"        lower whisker={s['lower_whisker']}",
            "    },",
            "] coordinates {",
        ]
        if outliers:
            for outlier in s["outliers"]:
                tikz.append(f"({idx}, {outlier})")
        tikz.append("};")

    tikz += [
        "\\end{axis}",
        "\\end{tikzpicture}",
    ]

    with Path(filepath).open(mode="w") as file_io:
        file_io.write("\n".join(tikz))


def main() -> None:
    """Main function to generate the LaTeX table from results data."""
    table_latex = summary_stats_to_latex()
    with Path("/tmp/table.tex").open(mode="w") as file_io:
        file_io.write(table_latex)

    for parameter, label in zip(["totals", "eigenvalues"], ["Total Time [\\si{\\second}]", "Eigenvalues"], strict=True):
        data = pd.read_csv(RESULTS_DIRECTORY / f"all_{parameter}.csv")
        stats = compute_boxplot_stats(data)
        tikz_from_boxplot_stats(stats, ylabel=label, filepath=f"/tmp/boxplot_{parameter}.tex")


if __name__ == "__main__":
    main()
