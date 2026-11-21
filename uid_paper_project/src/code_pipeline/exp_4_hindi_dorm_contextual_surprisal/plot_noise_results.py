#!/usr/bin/env python3
"""

himanshu - separated from noise output generation file.
This code file plots proportion bit loss for single and clustered noise and catastrophic counts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # File output; no GUI window is required.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None


DEFAULT_VERSION_ORDER = ["Uniform", "Actual", "Random", "Asymmetric"]
VALID_NOISE_TYPES = ("single", "clustered")
PLOT_WIDE_COLUMNS = [
    "type",
    "DORM",
    "single_proportion_bits_lost",
    "clustered_proportion_bits_lost",
    "single_catastrophic_loss",
    "clustered_catastrophic_loss",
]

DEFAULT_CHUNK_SIZE = 100_000

def validate_noise_type(noise_type: str) -> None:

    # we can raise an error if an unsupported noise type is supplied.
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(
            f"noise_type must be one of {VALID_NOISE_TYPES}, got {noise_type!r}."
        )


def sentence_version_plot_colors() -> Dict[str, str]:
    """Bar and raincloud colors matched to the supplied reference figure."""
    return {
        "Asymmetric": "#aae5f5",  # light cyan
        "Random": "#ebdfdd",      # pale pink-gray
        "Actual": "#ffb541",      # orange
        "Uniform": "#b3eac5",     # mint green
    }


def draw_density_box_points(
    ax: plt.Axes,
    values: np.ndarray,
    y_position: float,
    rng: np.random.Generator,
    color: str,
) -> None:
    
    """Draw one raincloud-like distribution row."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    jitter = rng.uniform(-0.10, 0.10, size=len(values))
    ax.scatter(
        values,
        np.full(len(values), y_position) + jitter,
        s=5,
        alpha=0.12,
        linewidths=0,
        color=color,
    )

    if gaussian_kde is not None and len(values) > 1 and np.std(values) > 0:
        x_min = max(0.0, float(np.percentile(values, 0.5)) - 0.05)
        x_max = min(1.0, float(np.percentile(values, 99.5)) + 0.05)

        if x_max <= x_min:
            x_min = max(0.0, float(values.min()) - 0.05)
            x_max = min(1.0, float(values.max()) + 0.05)

        x_grid = np.linspace(x_min, x_max, 300)
        kde = gaussian_kde(values)
        density = kde(x_grid)
        density = density / density.max() * 0.38

        ax.fill_between(
            x_grid,
            y_position,
            y_position + density,
            alpha=0.45,
            color=color,
        )
        ax.plot(
            x_grid,
            y_position + density,
            linewidth=1.3,
            color="black",
        )

    ax.boxplot(
        values,
        positions=[y_position],
        vert=False,
        widths=0.14,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        boxprops={
            "facecolor": "none",
            "edgecolor": "black",
            "linewidth": 1.1,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.7,
        },
        meanprops={
            "color": "#555079",
            "linestyle": ":",
            "linewidth": 2,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.1,
        },
        capprops={
            "color": "black",
            "linewidth": 0.0,
        },
    )

    # Mean marker, guide and value above the KDE curve (height <= 0.38).
    mean_value = float(np.mean(values))
    ax.vlines(mean_value, y_position - 0.10, y_position + 0.40,
              color="#555079", linestyle=":", linewidth=2, zorder=8)
    ax.scatter(mean_value, y_position + 0.22, s=32,
               color="black", zorder=10)
    ax.text(mean_value, y_position + 0.46, f"{mean_value:.2f}",
            ha="center", va="bottom", fontsize=9, zorder=10)


def plot_proportion_bit_loss_dist(
    wide_df: pd.DataFrame,
    noise_type: str,
    output_png: str,
) -> None:
    """
    Plot proportion-bit-loss distribution by sentence version using wide_df.
    """
    validate_noise_type(noise_type)

    print(f"Creating {noise_type} proportion-bit-loss distribution plot...",flush=True)

    proportion_column = f"{noise_type}_proportion_bits_lost"
    catastrophic_column = f"{noise_type}_catastrophic_loss"
    version_order = DEFAULT_VERSION_ORDER
    version_colors = sentence_version_plot_colors()
    catastrophic_counts = (
        pd.to_numeric(wide_df[catastrophic_column], errors="coerce")
        .fillna(0)
        .groupby(wide_df["type"], observed=False)
        .sum()
        .reindex(version_order, fill_value=0)
    )
    version_totals = (
        wide_df.groupby("type", observed=False)
        .size()
        .reindex(version_order, fill_value=0)
    )
    y_positions = list(range(len(version_order), 0, -1))
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(11, 6))

    for version, y_position in zip(version_order, y_positions):
        values = (
            pd.to_numeric(
                wide_df.loc[
                    wide_df["type"] == version,
                    proportion_column,
                ],
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        draw_density_box_points(ax, values, y_position, rng, color=version_colors[version])
        catastrophic_count = int(catastrophic_counts[version])
        sample_count = int(version_totals[version])
        catastrophic_pct = 100 * catastrophic_count / sample_count if sample_count else 0.0
        ax.text(
            1.02,
            y_position,
            f"Catastrophic: {catastrophic_count:,} ({catastrophic_pct:.2f}%)",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=10,
            clip_on=False,
        )

    # Losses of at least 50% are classified as catastrophic.
    ax.axvspan(0.50, 1.0, color="lightcoral", alpha=0.12, zorder=0)
    ax.axvline(0.50, color="firebrick", linestyle="--", linewidth=2, zorder=9)
    ax.text(0.52, 0.98, "50% loss threshold", transform=ax.transAxes,
            ha="left", va="top", color="firebrick", fontsize=10)
    ax.text(0.82, 0.91, "Catastrophic region", transform=ax.transAxes,
            ha="center", va="top", color="firebrick", fontsize=10)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(version_order, rotation=45, ha="right")
    ax.set_xlabel("Proportion of total bits lost", fontsize=13)
    ax.set_ylabel("Sentence version", fontsize=13)
    ax.set_title(f"{noise_type.title()} noise: proportion bits lost")
    ax.set_xlim(left=0, right=1)
    ax.grid(True, axis="both", alpha=0.25)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}", flush=True)

'''
# used to just experiment 

def plot_dorm_vs_bit_loss(
    wide_df: pd.DataFrame,
    noise_type: str,
    output_png: str,
) -> None:
    
    """Plot DORM against proportion bits lost using wide_df."""
    validate_noise_type(noise_type)

    print(f"Creating {noise_type} DORM-vs-bit-loss plot...", flush=True)

    proportion_column = f"{noise_type}_proportion_bits_lost"
    version_order = DEFAULT_VERSION_ORDER

    fig, ax = plt.subplots(figsize=(9, 6))

    for version in version_order:
        version_df = wide_df[wide_df["type"] == version]

        x_values = pd.to_numeric(
            version_df["DORM"],
            errors="coerce",
        ).to_numpy(dtype=float)

        y_values = pd.to_numeric(
            version_df[proportion_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]

        ax.scatter(
            x_values,
            y_values,
            s=6,
            alpha=0.12,
            linewidths=0,
            label=version,
        )

        if len(x_values) >= 2 and np.std(x_values) > 0:
            coefficients = np.polyfit(x_values, y_values, deg=1)
            x_grid = np.linspace(
                float(x_values.min()),
                float(x_values.max()),
                100,
            )
            y_grid = coefficients[0] * x_grid + coefficients[1]
            ax.plot(x_grid, y_grid, linewidth=1.5)

    ax.set_xlabel("DORM", fontsize=13)
    ax.set_ylabel("Proportion of total bits lost", fontsize=13)
    ax.set_title(f"{noise_type.title()} noise: DORM vs proportion bits lost")
    ax.set_ylim(bottom=0, top=1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}", flush=True)

'''

def plot_catastrophic_loss_comparison(
    wide_df: pd.DataFrame,
    output_png: str,
) -> None:

    """
    this function plots the number of catastrophic losses for each sentence version
    under single and clustered noise using wide_df
    """
    print("Creating catastrophic-loss comparison plot...", flush=True)

    version_order = DEFAULT_VERSION_ORDER
    version_colors = sentence_version_plot_colors()
    version_totals = (
        wide_df.groupby("type", observed=False)
        .size()
        .reindex(version_order, fill_value=0)
    )
    highest_count = 0.0

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        sharey=True,
    )

    noise_information = [
        ("single", "(A) Single Unit Noise"),
        ("clustered", "(B) Clustered Noise"),
    ]

    for ax, (noise_type, title) in zip(axes, noise_information):
        catastrophic_column = f"{noise_type}_catastrophic_loss"

        catastrophic_counts = (
            wide_df.assign(
                _catastrophic_value=pd.to_numeric(
                    wide_df[catastrophic_column],
                    errors="coerce",
                ).fillna(0)
            )
            .groupby("type", observed=False)["_catastrophic_value"]
            .sum()
            .reindex(version_order, fill_value=0)
        )

        bars = ax.bar(
            catastrophic_counts.index,
            catastrophic_counts.values,
            width=0.65,
            color=[version_colors[version] for version in catastrophic_counts.index],
        )
        ax.bar_label(
            bars,
            labels=[
                f"{int(count):,} ({100 * count / version_totals[version]:.2f}%)"
                if version_totals[version] else "0 (0.00%)"
                for version, count in catastrophic_counts.items()
            ],
            padding=4,
            fontsize=9,
        )
        highest_count = max(highest_count, float(catastrophic_counts.max()))
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Sentence Version", fontsize=12)

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Number of Catastrophic Losses", fontsize=13)
    axes[0].set_ylim(0, max(1.0, highest_count * 1.15))

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}", flush=True)



# read only the plotting columns from the existing output CSV.

def load_compact_plot_dataframe(input_csv, chunk_size = DEFAULT_CHUNK_SIZE) -> pd.DataFrame:
   
    # selected columns, we will  keep only a small DataFrame
    
    input_path = Path(input_csv)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    numeric_dtype = "float32"
    frames: list[pd.DataFrame] = []
    total_rows = 0
    print(f"Reading plotting columns from: {input_path}", flush=True)

    reader = pd.read_csv(
        input_path,
        usecols=PLOT_WIDE_COLUMNS,
        chunksize=DEFAULT_CHUNK_SIZE,
        dtype={"type": "string"},
        encoding="utf-8",
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        chunk["type"] = pd.Categorical(
            chunk["type"], categories=DEFAULT_VERSION_ORDER
        )
        for column in PLOT_WIDE_COLUMNS[1:]:
            chunk[column] = pd.to_numeric(
                chunk[column], errors="coerce"
            ).astype(numeric_dtype)

        frames.append(chunk)
        total_rows += len(chunk)

        if chunk_number == 1 or chunk_number % 10 == 0:
            print(f"  Read {total_rows:,} rows ({chunk_number} chunks)", flush=True)

    if not frames:
        raise ValueError("The input CSV contains no data rows.")

    print("Combining compact plotting data...", flush=True)
    wide_df = pd.concat(frames, ignore_index=True)
    frames.clear()
    wide_df["type"] = pd.Categorical(
        wide_df["type"], categories=DEFAULT_VERSION_ORDER
    )

    megabytes = wide_df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(
        f"Loaded {len(wide_df):,} rows; compact DataFrame is "
        f"approximately {megabytes:,.1f} MiB.",
        flush=True,
    )
    print("Rows per sentence version:", flush=True)
    print(wide_df["type"].value_counts(sort=False).to_string(), flush=True)
    return wide_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Make noise-loss figures from an existing output CSV ")
    parser.add_argument("--input_csv", required=True, help="Existing noise_output.csv path.")

    args = parser.parse_args()
    
    wide_df = load_compact_plot_dataframe(args.input_csv, chunk_size=DEFAULT_CHUNK_SIZE)

    hindi_single_noise_png = ("./outputs/mtp_2_features/exp_4_demo_output/noise_plots/bit_loss_single_noise.png")
    hindi_clustered_noise_png = ("./outputs/mtp_2_features/exp_4_demo_output/noise_plots/bit_loss_clustered_noise.png")
    hindi_catastrophic_png = ("./outputs/mtp_2_features/exp_4_demo_output/noise_plots/catastrophic_loss_comparison.png")
 

    plot_catastrophic_loss_comparison(wide_df, output_png = hindi_catastrophic_png)
  
    plot_proportion_bit_loss_dist(wide_df, "single", output_png = hindi_single_noise_png)

    plot_proportion_bit_loss_dist(wide_df, "clustered", output_png = hindi_clustered_noise_png)
    '''
    # used as an experiment
    for noise_type in VALID_NOISE_TYPES:

        plot_dorm_vs_bit_loss(wide_df, noise_type, output_png = hindi_dorm_bit_loss_png)

    print("Finished plotting !", flush=True)
    '''

if __name__ == "__main__":
    main()
