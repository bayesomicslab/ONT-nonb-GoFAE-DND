#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SPLITS = ["train", "val", "test"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def load_all_splits(folder):
    data_parts = []
    label_parts = []
    true_label_parts = []

    feature_cols = [f"dwell_{i}" for i in range(50)]

    for split in SPLITS:
        csv_path = folder / f"same_{split}_normalized.csv"
        label_path = folder / f"same_{split}_label.npy"
        true_label_path = folder / f"same_{split}_true_label.npy"

        for path in [csv_path, label_path, true_label_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")

        df = pd.read_csv(csv_path)

        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{csv_path.name} does not contain the expected dwell_0 ... dwell_49 columns."
            )

        x = df[feature_cols].to_numpy(dtype=np.float32)
        label = np.load(label_path).astype(np.uint8)
        true_label = np.load(true_label_path).astype(np.uint8)

        if not (len(x) == len(label) == len(true_label)):
            raise ValueError(
                f"Row mismatch in {split}: "
                f"CSV={len(x)}, label={len(label)}, true_label={len(true_label)}"
            )

        if not np.isin(label, [0, 1]).all():
            raise ValueError(f"{label_path.name} contains values other than 0 and 1.")

        if not np.isin(true_label, [0, 1]).all():
            raise ValueError(f"{true_label_path.name} contains values other than 0 and 1.")

        # Verify that the observed binary label vector matches the CSV label column.
        csv_binary_label = (df["label"].astype(str) != "bdna").astype(np.uint8).to_numpy()
        if not np.array_equal(csv_binary_label, label):
            raise ValueError(
                f"Observed-label mismatch in {split}: "
                f"{csv_path.name} and {label_path.name} are not row-aligned."
            )

        data_parts.append(x)
        label_parts.append(label)
        true_label_parts.append(true_label)

    # Stack train + validation + test.
    data = np.vstack(data_parts)
    label = np.concatenate(label_parts)
    true_label = np.concatenate(true_label_parts)

    return data, label, true_label


def print_label_summary(label, true_label):
    print("\nAll splits combined")
    print("-------------------")
    print(f"Total samples: {len(label):,}")
    print(f"Observed B-DNA (label=0):     {(label == 0).sum():,}")
    print(f"Observed non-B (label=1):     {(label == 1).sum():,}")
    print(f"True B-DNA (true_label=0):    {(true_label == 0).sum():,}")
    print(f"True non-B (true_label=1):    {(true_label == 1).sum():,}")

    print("\nObserved label vs. true label")
    print("-----------------------------")
    print("                         true B-DNA   true non-B")
    print(
        f"observed B-DNA (0)      "
        f"{np.sum((label == 0) & (true_label == 0)):>11,}   "
        f"{np.sum((label == 0) & (true_label == 1)):>10,}"
    )
    print(
        f"observed non-B (1)      "
        f"{np.sum((label == 1) & (true_label == 0)):>11,}   "
        f"{np.sum((label == 1) & (true_label == 1)):>10,}"
    )


def plot_true_label_quantiles(folder):
    data, label, true_label = load_all_splits(folder)

    # The plot is defined ONLY by the ground-truth class.
    true_bdna = data[true_label == 0]
    true_nonb = data[true_label == 1]

    if len(true_bdna) == 0:
        raise ValueError("There are no true B-DNA samples (true_label=0).")
    if len(true_nonb) == 0:
        raise ValueError("There are no true non-B samples (true_label=1).")

    print_label_summary(label, true_label)

    x = np.arange(50)

    fig, ax = plt.subplots(figsize=(10, 7))

    for quantile, color in zip(QUANTILES, COLORS):
        nonb_quantile = np.quantile(true_nonb, quantile, axis=0)
        bdna_quantile = np.quantile(true_bdna, quantile, axis=0)

        # Solid: true non-B
        ax.plot(
            x,
            nonb_quantile,
            color=color,
            linewidth=3,
        )

        # Dashed: true B-DNA control
        ax.plot(
            x,
            bdna_quantile,
            color=color,
            linestyle="--",
            linewidth=3,
        )

    ax.set_xlabel("Position in Window", fontsize=18)
    ax.set_ylabel("Translocation Time", fontsize=18)
    ax.set_title("G Quadruplex Motif (Synthetic Data) - Forward", fontsize=20)

    quantile_handles = [
        Line2D([0], [0], color=color, linewidth=3, label=str(q))
        for q, color in zip(QUANTILES, COLORS)
    ]
    quantile_legend = ax.legend(
        handles=quantile_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.50),
        title="Quantile",
        prop={"size": 14},
        title_fontsize=15,
    )
    ax.add_artist(quantile_legend)

    structure_handles = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=3),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=3),
    ]
    ax.legend(
        handles=structure_handles,
        labels=["True non-B DNA", "True B-DNA (Control)"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.82),
        title="Ground Truth",
        prop={"size": 14},
        title_fontsize=15,
    )

    plt.tight_layout()

    output_path = folder / "G_Quadruplex_Motif_true_label_quantiles.png"
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stack train/val/test simulated G-quadruplex data and plot "
            "quantiles for TRUE B-DNA versus TRUE non-B."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help=(
            "Folder containing same_train_normalized.csv, "
            "same_val_normalized.csv, same_test_normalized.csv and "
            "their *_label.npy and *_true_label.npy files."
        ),
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {args.folder}")

    plot_true_label_quantiles(args.folder)


if __name__ == "__main__":
    main()

