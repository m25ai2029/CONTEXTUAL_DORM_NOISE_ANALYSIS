
"""
Plot raincloud curves of sqrt(DORM) for four sentence versions.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import gaussian_kde

DEFAULT_VERSION_ORDER = ["Uniform", "Actual", "Random", "Asymmetric"]
VERSION_COLORS = {
    "Uniform": "#b3eac5",
    "Actual": "#ffb541",
    "Random": "#ebdfdd",
    "Asymmetric": "#aae5f5",
}


def normalise_version_name(value: str) -> str:
    """Map common labels to the four labels used in the plot."""
    value = str(value).strip().lower()

    mapping = {
        "uniform": "Uniform",
        "uido": "Uniform",
        "uido_uniform": "Uniform",
        "actual": "Actual",
        "reference": "Actual",
        "ref": "Actual",
        "original": "Actual",
        "random": "Random",
        "rand": "Random",
        "asymmetric": "Asymmetric",
        "asym": "Asymmetric",
    }

    return mapping.get(value, str(value).strip().title())


def clean_dataframe(df: pd.DataFrame, version_column: str, dorm_column: str) -> pd.DataFrame:

    """Keep only required columns and create sqrt(DORM)."""
    clean_df = df[[version_column, dorm_column]].copy()
    clean_df[version_column] = clean_df[version_column].apply(normalise_version_name)
    clean_df[dorm_column] = pd.to_numeric(clean_df[dorm_column], errors="coerce")
    clean_df = clean_df.dropna(subset=[version_column, dorm_column])
    clean_df = clean_df[clean_df[dorm_column] >= 0]
    clean_df["sqrt_dorm"] = np.sqrt(clean_df[dorm_column])
    return clean_df


def draw_one_raincloud(
    ax: plt.Axes,
    values: np.ndarray,
    y_position: float,
    label: str,
    rng: np.random.Generator,
) -> None:
    """Draw a raincloud with clearly distinguished mean and median lines."""
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    color = VERSION_COLORS[label]

    # Jittered raw points,  # Creates small random vertical movement so dots do not overlap exactly
    jitter = rng.uniform(-0.12, 0.12, size=len(values))
    ax.scatter(
        values,
        np.full(len(values), y_position) + jitter,
        s=5,
        alpha=0.12,
        linewidths=0,
        color=color,
    )

    # All plotted summaries use these same sqrt(DORM) values.
    mean_value = float(np.mean(values))
    sd_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    median_value = float(np.median(values))
    print(f"{label}: mean(sqrt(DORM)) = {mean_value:.4f}; "
          f"median(sqrt(DORM)) = {median_value:.4f}")
    # KDE cloud above each row
    if len(values) > 1 and np.std(values) > 0:      # values should not be identical
        x_min = max(0.0, float(np.percentile(values, 0.5)) - 0.35)
        x_max = float(np.percentile(values, 99.5)) + 0.35
        x_grid = np.linspace(x_min, x_max, 300)          # creates 300 points on x axis
        kde = gaussian_kde(values)          # estimates probability density
        density = kde(x_grid)
        density = density / density.max() * 0.45            # Normalizes the density height so it fits nicely in plot

        # Fills the KDE cloud area
        ax.fill_between(
            x_grid,
            y_position,
            y_position + density,
            alpha=0.55,
            color=color,
        )
        ax.plot(x_grid, y_position + density, linewidth=1.5, color="black")

    # The solid black line is the median. The dashed brown line is the mean.
    # Matplotlib does not show a boxplot mean unless showmeans=True.
    ax.boxplot(
        values,
        positions=[y_position],
        vert=False,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.8},
        meanprops={"color": "firebrick", "linestyle": "--", "linewidth": 2},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 0.0},
    )

    # Draw this LAST: the KDE and box must not hide the mean guide.
    # It matches the mean drawn inside the boxplot exactly.
    ax.vlines(mean_value, y_position - 0.12, y_position + 0.40,
              linestyle="--", linewidth=2, color="firebrick", zorder=8)
    ax.scatter(mean_value, y_position + 0.20, marker="o", s=30,
               color="black", zorder=10)
    ax.text(mean_value, y_position + 0.50, f"{mean_value:.2f} ± {sd_value:.2f}",
            ha="center", va="bottom",fontsize=8, zorder=10)


def plot_raincloud(
    df: pd.DataFrame,
    version_column: str,
    output_png: str,
    title: str | None = None,
) -> None:
    
    """Create raincloud plot similar to the attached paper figure."""
    rng = np.random.default_rng(42)

    available_versions = list(df[version_column].dropna().unique())
    version_order = [v for v in DEFAULT_VERSION_ORDER if v in available_versions]
    #remaining_versions = [v for v in available_versions if v not in version_order]
    #version_order.extend(sorted(remaining_versions))

    y_positions = list(range(len(version_order), 0, -1))

    fig, ax = plt.subplots(figsize=(11, 6))

    for version, y_position in zip(version_order, y_positions):
        values = df.loc[df[version_column] == version, "sqrt_dorm"].to_numpy()
        draw_one_raincloud(ax, values, y_position, version, rng)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(version_order, rotation=45, ha="right")
    ax.set_xlabel(r"$\sqrt{DORM}$", fontsize=14)
    ax.set_ylabel("Sentence Version", fontsize=14)

    if title:
        ax.set_title(title)

    ax.legend(
        handles=[
            Line2D([0], [0], color="firebrick", linestyle="--", linewidth=2,
                   label="Mean of sqrt(DORM)"),
            Line2D([0], [0], color="black", linewidth=1.8,
                   label="Median (boxplot)"),
        ],
        loc="upper right",
        frameon=False,
    )

    ax.grid(True, axis="both", alpha=0.25)
    ax.set_axisbelow(True)

    x_max = df["sqrt_dorm"].quantile(0.995)
    if pd.notna(x_max):
        ax.set_xlim(left=0, right=max(1.0, float(x_max) + 0.5))

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raincloud distributions of sqrt(DORM) for four sentence versions."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV file.")
    parser.add_argument("--output_png", required=True, help="Output PNG path.")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv,encoding="utf-8",dtype=str)

    version_column =  "type"
    dorm_column = "DORM"

    clean_df = clean_dataframe(df, version_column, dorm_column)

    print("Input rows:", len(df))
    print("Rows used for plotting:", len(clean_df))
    print("Version column:", version_column)
    print("DORM column:", dorm_column)
    print("Counts by version:")
    print(clean_df[version_column].value_counts().to_string())

    plot_raincloud(
        clean_df,
        version_column = version_column,
        output_png = args.output_png,
        title = "Sentence Version vs DORM Plot",
    )

    print("Plot saved to:", args.output_png)


if __name__ == "__main__":
    main()
