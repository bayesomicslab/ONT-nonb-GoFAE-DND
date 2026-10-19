#!/usr/bin/env python3

import argparse
import gc
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


AMPLITUDES = np.array([2.0, 2.5, 3.0], dtype=np.float32)
# AMPLITUDES = np.array([0.5, 0.75, 1.0], dtype=np.float32)
MOTIF_LENGTHS = np.array([5, 10, 20], dtype=np.int16)
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
QUANTILE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
SPLITS = ["train", "val", "test"]
FEATURE_COLUMNS = [f"dwell_{i}" for i in range(50)]

TYPE_ALIASES = {
    "g4": "G_Quadruplex_Motif",
    "gquad": "G_Quadruplex_Motif",
    "g_quadruplex": "G_Quadruplex_Motif",
    "g_quadruplex_motif": "G_Quadruplex_Motif",
    "g-quadruplex": "G_Quadruplex_Motif",
    "str": "Short_tandem_repeat",
    "short_tandem_repeat": "Short_tandem_repeat",
    "short-tandem-repeat": "Short_tandem_repeat",
}


# -----------------------------------------------------------------------------
# Arguments / names
# -----------------------------------------------------------------------------

def ratio(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1")
    return value


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def canonical_nonb_type(value):
    key = value.strip().lower()
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]
    if value in {"G_Quadruplex_Motif", "Short_tandem_repeat"}:
        return value
    raise argparse.ArgumentTypeError(
        "non-B type must be G_Quadruplex_Motif/g4 or "
        "Short_tandem_repeat/str"
    )


def display_name(nonb_type):
    if nonb_type == "G_Quadruplex_Motif":
        return "G Quadruplex Motif (Synthetic Data)"
    if nonb_type == "Short_tandem_repeat":
        return "Short Tandem Repeat (Synthetic Data)"
    return nonb_type.replace("_", " ") + " (Synthetic Data)"


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def exact_binary_mask(n, fraction_true, rng):
    """Return exactly round(n*fraction_true) true entries, randomly located."""
    n_true = int(round(n * fraction_true))
    mask = np.zeros(n, dtype=bool)
    mask[:n_true] = True
    rng.shuffle(mask)
    return mask


def make_nonb_pattern(nonb_type, motif_length, amplitude):
    # Keep the simulator's existing t construction; only the G4 equations below
    # are corrected to the paper-code form requested by the user.
    t = np.linspace(-25.0, 25.0, motif_length + 2, dtype=np.float32)[1:-1]

    if nonb_type == "G_Quadruplex_Motif":
        a = np.float32(0.05)
        b = np.float32(amplitude)

        # Paper-code form:
        # forward: upward / inverted quadratic around the motif center
        # reverse: corresponding opposite direction
        forward = -((a * t) ** 2) + b
        reverse = ((a * t) ** 2) - b

    elif nonb_type == "Short_tandem_repeat":
        c = np.float32(0.3)
        signal = np.float32(amplitude) * np.sin(c * t)
        forward = signal
        reverse = signal.copy()

    else:
        raise ValueError(f"Unsupported non-B type: {nonb_type}")

    return np.stack([forward, reverse], axis=0).astype(np.float32)


def add_nonb_signals(x, true_is_nonb, motif_lengths, amplitudes, nonb_type):
    true_indices = np.flatnonzero(true_is_nonb)

    for motif_length in MOTIF_LENGTHS:
        for amplitude in AMPLITUDES:
            selected = true_indices[
                (motif_lengths[true_indices] == motif_length)
                & np.isclose(amplitudes[true_indices], amplitude)
            ]

            if selected.size == 0:
                continue

            pattern = make_nonb_pattern(
                nonb_type,
                int(motif_length),
                float(amplitude),
            )

            start = (x.shape[2] - int(motif_length)) // 2
            end = start + int(motif_length)
            x[selected, :, start:end] += pattern[None, :, :]


def generate_observed_class(
    output_file,
    n_samples,
    p_true_nonb,
    nonb_type,
    window_size,
    noise_sd,
    rng,
    chunk_size,
):
    """
    Generate one observed-label class.

    p_true_nonb means:
      P(true non-B | this observed label)

    Returns:
      temporary (N, 2, window_size) .npy path
      true labels: 0 = true B-DNA, 1 = true non-B
    """
    true_is_nonb = exact_binary_mask(n_samples, p_true_nonb, rng)

    motif_lengths = np.zeros(n_samples, dtype=np.int16)
    amplitudes = np.zeros(n_samples, dtype=np.float32)

    true_indices = np.flatnonzero(true_is_nonb)
    if true_indices.size:
        motif_lengths[true_indices] = rng.choice(
            MOTIF_LENGTHS, size=true_indices.size
        )
        amplitudes[true_indices] = rng.choice(
            AMPLITUDES, size=true_indices.size
        )

    data = np.lib.format.open_memmap(
        output_file,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, 2, window_size),
    )

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        chunk = rng.normal(
            0.0,
            noise_sd,
            size=(end - start, 2, window_size),
        ).astype(np.float32)

        add_nonb_signals(
            chunk,
            true_is_nonb[start:end],
            motif_lengths[start:end],
            amplitudes[start:end],
            nonb_type,
        )

        data[start:end] = chunk

    data.flush()
    del data

    return Path(output_file), true_is_nonb.astype(np.uint8)


# -----------------------------------------------------------------------------
# Split / save
# -----------------------------------------------------------------------------

def split_indices(n, train_ratio, val_ratio, rng):
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def make_split_part(data, true_labels, indices, observed_is_nonb, nonb_type):
    # Final dataset uses only the forward strand.
    forward = data[indices, 0, :]

    # Final dataset uses the middle 50 bases.
    start = (forward.shape[1] - 50) // 2
    x = np.asarray(forward[:, start:start + 50], dtype=np.float32)

    # Binary observed label used by the method.
    observed_binary = np.full(
        len(indices),
        1 if observed_is_nonb else 0,
        dtype=np.uint8,
    )

    observed_string = np.full(
        len(indices),
        nonb_type if observed_is_nonb else "bdna",
        dtype=object,
    )

    # Ground-truth class from the simulator.
    true_binary = true_labels[indices].astype(np.uint8)

    return x, observed_string, observed_binary, true_binary


def save_split(type_dir, split_name, parts, rng):
    # First combine the observed B-DNA and observed non-B blocks.
    x = np.concatenate([p[0] for p in parts], axis=0)
    string_labels = np.concatenate([p[1] for p in parts], axis=0)
    label_binary = np.concatenate([p[2] for p in parts], axis=0)
    true_label_binary = np.concatenate([p[3] for p in parts], axis=0)

    # Then shuffle ROWS ONLY. The exact same row permutation is applied to
    # features, observed labels, and true labels. Columns remain unchanged.
    permutation = rng.permutation(len(x))
    x = x[permutation]
    string_labels = string_labels[permutation]
    label_binary = label_binary[permutation]
    true_label_binary = true_label_binary[permutation]

    df = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    df["label"] = string_labels

    df.to_csv(type_dir / f"same_{split_name}_normalized.csv", index=False)

    np.save(type_dir / f"same_{split_name}_label.npy", label_binary)
    np.save(type_dir / f"same_{split_name}_true_label.npy", true_label_binary)

    np.savez_compressed(
        type_dir / f"same_{split_name}_label.npz",
        label=label_binary,
    )
    np.savez_compressed(
        type_dir / f"same_{split_name}_true_label.npz",
        true_label=true_label_binary,
    )

    print(
        f"{split_name}: {len(x):,} samples | "
        f"label 0={np.sum(label_binary == 0):,}, "
        f"label 1={np.sum(label_binary == 1):,} | "
        f"true 0={np.sum(true_label_binary == 0):,}, "
        f"true 1={np.sum(true_label_binary == 1):,}"
    )


# -----------------------------------------------------------------------------
# Loading / validation for plotting and normalization
# -----------------------------------------------------------------------------

def load_split(dataset_dir, split_name):
    csv_path = dataset_dir / f"same_{split_name}_normalized.csv"
    label_path = dataset_dir / f"same_{split_name}_label.npy"
    true_path = dataset_dir / f"same_{split_name}_true_label.npy"

    df = pd.read_csv(csv_path)
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    label = np.load(label_path).astype(np.uint8)
    true_label = np.load(true_path).astype(np.uint8)

    if not (len(df) == len(label) == len(true_label)):
        raise ValueError(
            f"Row mismatch in {dataset_dir}/{split_name}: "
            f"CSV={len(df)}, label={len(label)}, true_label={len(true_label)}"
        )

    csv_binary = (df["label"].astype(str) != "bdna").astype(np.uint8).to_numpy()
    if not np.array_equal(csv_binary, label):
        raise ValueError(
            f"Observed label mismatch in {split_name}: CSV and label.npy are not aligned."
        )

    return df, x, label, true_label


def load_all_splits(dataset_dir):
    xs = []
    labels = []
    true_labels = []

    for split_name in SPLITS:
        _, x, label, true_label = load_split(dataset_dir, split_name)
        xs.append(x)
        labels.append(label)
        true_labels.append(true_label)

    return (
        np.vstack(xs),
        np.concatenate(labels),
        np.concatenate(true_labels),
    )


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

def copy_label_files(source_dir, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)

    for split_name in SPLITS:
        for suffix in [
            "label.npy",
            "true_label.npy",
            "label.npz",
            "true_label.npz",
        ]:
            src = source_dir / f"same_{split_name}_{suffix}"
            dst = target_dir / src.name
            shutil.copy2(src, dst)


def save_mean_std_normalized(source_dir, target_dir):
    """
    GLOBAL scalar mean/std normalization.

    Compute ONE mean and ONE standard deviation using ALL dwell values from
    OBSERVED B-DNA (label == 0) samples in the TRAIN split. The same two
    scalar values are then applied to every dwell value in train/val/test.
    """
    _, train_x, train_label, _ = load_split(source_dir, "train")
    bdna_train = train_x[train_label == 0]

    if len(bdna_train) == 0:
        raise ValueError("No observed B-DNA samples in training split.")

    # No axis -> one scalar over all B-DNA training samples and all 50 positions.
    mean = float(np.mean(bdna_train))
    std = float(np.std(bdna_train, ddof=0))

    if std == 0:
        std = 1.0

    target_dir.mkdir(parents=True, exist_ok=True)
    copy_label_files(source_dir, target_dir)

    for split_name in SPLITS:
        df, x, _, _ = load_split(source_dir, split_name)
        x_norm = (x - mean) / std

        out_df = pd.DataFrame(x_norm, columns=FEATURE_COLUMNS)
        out_df["label"] = df["label"].to_numpy()
        out_df.to_csv(
            target_dir / f"same_{split_name}_normalized.csv",
            index=False,
        )

    print(f"Mean/std normalization scalars: mean={mean:.8f}, std={std:.8f}")


def save_median_iqr_normalized(source_dir, target_dir):
    """
    GLOBAL scalar median/IQR normalization.

    Compute ONE median and ONE IQR using ALL dwell values from OBSERVED B-DNA
    (label == 0) samples in the TRAIN split. The same two scalar values are
    then applied to every dwell value in train/val/test.
    """
    _, train_x, train_label, _ = load_split(source_dir, "train")
    bdna_train = train_x[train_label == 0]

    if len(bdna_train) == 0:
        raise ValueError("No observed B-DNA samples in training split.")

    # No axis -> one scalar over all B-DNA training samples and all 50 positions.
    median = float(np.median(bdna_train))
    q25, q75 = np.percentile(bdna_train, [25, 75])
    iqr = float(q75 - q25)

    if iqr == 0:
        iqr = 1.0

    target_dir.mkdir(parents=True, exist_ok=True)
    copy_label_files(source_dir, target_dir)

    for split_name in SPLITS:
        df, x, _, _ = load_split(source_dir, split_name)
        x_norm = (x - median) / iqr

        out_df = pd.DataFrame(x_norm, columns=FEATURE_COLUMNS)
        out_df["label"] = df["label"].to_numpy()
        out_df.to_csv(
            target_dir / f"same_{split_name}_normalized.csv",
            index=False,
        )

    print(f"Median/IQR normalization scalars: median={median:.8f}, IQR={iqr:.8f}")


# -----------------------------------------------------------------------------
# Quantile plots
# -----------------------------------------------------------------------------

def plot_quantiles(
    dataset_dir,
    nonb_type,
    grouping,
    output_filename,
    ylabel,
):
    """
    grouping='true'     -> group by true_label.npy
    grouping='observed' -> group by label.npy
    """
    data, label, true_label = load_all_splits(dataset_dir)

    if grouping == "true":
        grouping_vector = true_label
        legend_title = "Ground Truth"
        nonb_legend = "True non-B DNA"
        bdna_legend = "True B-DNA (Control)"
        title_suffix = "True Labels"
    elif grouping == "observed":
        grouping_vector = label
        legend_title = "Observed Label"
        nonb_legend = "Non-B Label"
        bdna_legend = "B-DNA Label (Control)"
        title_suffix = "Observed Labels"
    else:
        raise ValueError("grouping must be 'true' or 'observed'")

    bdna = data[grouping_vector == 0]
    nonb = data[grouping_vector == 1]

    if len(bdna) == 0 or len(nonb) == 0:
        raise ValueError(
            f"Cannot make {grouping} plot: one of the two groups is empty."
        )

    x_axis = np.arange(data.shape[1])
    fig, ax = plt.subplots(figsize=(10, 7))

    for q, color in zip(QUANTILES, QUANTILE_COLORS):
        nonb_q = np.quantile(nonb, q, axis=0)
        bdna_q = np.quantile(bdna, q, axis=0)

        ax.plot(
            x_axis,
            nonb_q,
            color=color,
            linewidth=3,
        )
        ax.plot(
            x_axis,
            bdna_q,
            color=color,
            linestyle="--",
            linewidth=3,
        )

    ax.set_xlabel("Position in Window", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(
        f"{display_name(nonb_type)} - Forward\nGrouped by {title_suffix}",
        fontsize=20,
    )

    quantile_handles = [
        Line2D([0], [0], color=color, linewidth=3, label=str(q))
        for q, color in zip(QUANTILES, QUANTILE_COLORS)
    ]
    quantile_legend = ax.legend(
        handles=quantile_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.48),
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
        labels=[nonb_legend, bdna_legend],
        loc="center left",
        bbox_to_anchor=(1.02, 0.82),
        title=legend_title,
        prop={"size": 14},
        title_fontsize=15,
    )

    plt.tight_layout()
    plt.savefig(
        dataset_dir / output_filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_all_plots(dataset_dir, nonb_type, ylabel):
    plot_quantiles(
        dataset_dir=dataset_dir,
        nonb_type=nonb_type,
        grouping="true",
        output_filename="quantiles_true_labels.png",
        ylabel=ylabel,
    )
    plot_quantiles(
        dataset_dir=dataset_dir,
        nonb_type=nonb_type,
        grouping="observed",
        output_filename="quantiles_observed_labels.png",
        ylabel=ylabel,
    )


# -----------------------------------------------------------------------------
# Main simulation workflow
# -----------------------------------------------------------------------------

def simulate_one_type(
    outdir,
    nonb_type,
    n_nonb,
    n_b,
    nonb_ratio,
    b_ratio,
    window_size,
    noise_sd,
    seed,
    chunk_size,
):
    if window_size < 50:
        raise ValueError("window size must be at least 50")

    type_dir = outdir / nonb_type
    type_dir.mkdir(parents=True, exist_ok=True)

    seed_sequence = np.random.SeedSequence(seed)
    nonb_seed, b_seed, split_seed = seed_sequence.spawn(3)

    rng_nonb = np.random.default_rng(nonb_seed)
    rng_b = np.random.default_rng(b_seed)
    rng_split = np.random.default_rng(split_seed)

    # Temporary raw 2 x 100 arrays are used only to construct the final
    # forward/central-50 datasets. They are removed after the split is saved.
    with tempfile.TemporaryDirectory(
        prefix=".simulator2_tmp_",
        dir=type_dir,
        ignore_cleanup_errors=True,
    ) as tmp:
        tmp = Path(tmp)

        nonb_file, nonb_true = generate_observed_class(
            output_file=tmp / "observed_nonb.npy",
            n_samples=n_nonb,
            p_true_nonb=nonb_ratio,
            nonb_type=nonb_type,
            window_size=window_size,
            noise_sd=noise_sd,
            rng=rng_nonb,
            chunk_size=chunk_size,
        )

        b_file, b_true = generate_observed_class(
            output_file=tmp / "observed_b.npy",
            n_samples=n_b,
            p_true_nonb=b_ratio,
            nonb_type=nonb_type,
            window_size=window_size,
            noise_sd=noise_sd,
            rng=rng_b,
            chunk_size=chunk_size,
        )

        nonb_data = np.load(nonb_file, mmap_mode="r")
        b_data = np.load(b_file, mmap_mode="r")

        # Paper split:
        # observed B-DNA: 30% train / 20% validation / 50% test
        b_train, b_val, b_test = split_indices(n_b, 0.30, 0.20, rng_split)

        # observed non-B: 10% train / 10% validation / 80% test
        nonb_train, nonb_val, nonb_test = split_indices(
            n_nonb, 0.10, 0.10, rng_split
        )

        train_parts = [
            make_split_part(
                b_data, b_true, b_train,
                observed_is_nonb=False,
                nonb_type=nonb_type,
            ),
            make_split_part(
                nonb_data, nonb_true, nonb_train,
                observed_is_nonb=True,
                nonb_type=nonb_type,
            ),
        ]

        val_parts = [
            make_split_part(
                b_data, b_true, b_val,
                observed_is_nonb=False,
                nonb_type=nonb_type,
            ),
            make_split_part(
                nonb_data, nonb_true, nonb_val,
                observed_is_nonb=True,
                nonb_type=nonb_type,
            ),
        ]

        test_parts = [
            make_split_part(
                b_data, b_true, b_test,
                observed_is_nonb=False,
                nonb_type=nonb_type,
            ),
            make_split_part(
                nonb_data, nonb_true, nonb_test,
                observed_is_nonb=True,
                nonb_type=nonb_type,
            ),
        ]

        save_split(type_dir, "train", train_parts, rng_split)
        save_split(type_dir, "val", val_parts, rng_split)
        save_split(type_dir, "test", test_parts, rng_split)

        # Explicitly close mmap handles before deleting temporary files.
        if isinstance(nonb_data, np.memmap) and nonb_data._mmap is not None:
            nonb_data._mmap.close()
        if isinstance(b_data, np.memmap) and b_data._mmap is not None:
            b_data._mmap.close()

        del nonb_data
        del b_data
        gc.collect()

    # 1) Plots of the unscaled/original final dataset.
    make_all_plots(
        dataset_dir=type_dir,
        nonb_type=nonb_type,
        ylabel="Translocation Time",
    )

    # 2) Mean/std normalization using ONLY observed B-DNA in TRAIN.
    mean_std_dir = type_dir / "mean_std_normalized"
    save_mean_std_normalized(type_dir, mean_std_dir)
    make_all_plots(
        dataset_dir=mean_std_dir,
        nonb_type=nonb_type,
        ylabel="Normalized Translocation Time",
    )

    # 3) Median/IQR normalization using ONLY observed B-DNA in TRAIN.
    median_iqr_dir = type_dir / "median_iqr_normalized"
    save_median_iqr_normalized(type_dir, median_iqr_dir)
    make_all_plots(
        dataset_dir=median_iqr_dir,
        nonb_type=nonb_type,
        ylabel="Normalized Translocation Time",
    )

    print(f"\nSaved final dataset to: {type_dir}")
    print(f"Mean/std normalized dataset: {mean_std_dir}")
    print(f"Median/IQR normalized dataset: {median_iqr_dir}")
    print("\nBinary encoding:")
    print("  0 = B-DNA / inlier")
    print("  1 = non-B / outlier")
    print("\nNormalization reference:")
    print("  statistics are computed from OBSERVED B-DNA (label=0) in TRAIN only")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-nb",
        "--num-nonb",
        dest="n_nonb",
        type=positive_int,
        default=20000,
    )

    parser.add_argument(
        "-b",
        "--num-b",
        dest="n_b",
        type=positive_int,
        default=200000,
    )

    parser.add_argument(
        "-r",
        "--nonb-ratio",
        "--nonb_ratio",
        dest="nonb_ratio",
        type=ratio,
        default=0.10,
        help="P(true non-B | labeled non-B)",
    )

    parser.add_argument(
        "--b-ratio",
        "--b_ratio",
        dest="b_ratio",
        type=ratio,
        default=0.0,
        help="P(true non-B | labeled B-DNA)",
    )

    parser.add_argument(
        "-t",
        "--nonb-type",
        "--nonb_type",
        dest="nonb_type",
        default="G_Quadruplex_Motif",
    )

    parser.add_argument(
        "-s",
        "--save-path",
        "--output",
        dest="output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--window-size",
        type=positive_int,
        default=100,
    )

    parser.add_argument(
        "--noise-sd",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
    )

    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=10000,
    )

    return parser


def main():
    args = build_parser().parse_args()

    if args.noise_sd <= 0:
        raise SystemExit("--noise-sd must be > 0")

    if args.window_size < int(MOTIF_LENGTHS.max()):
        raise SystemExit(
            f"--window-size must be at least {int(MOTIF_LENGTHS.max())}"
        )

    requested = args.nonb_type.strip().lower()

    if requested == "both":
        nonb_types = [
            "G_Quadruplex_Motif",
            "Short_tandem_repeat",
        ]
    else:
        nonb_types = [canonical_nonb_type(args.nonb_type)]

    args.output.mkdir(parents=True, exist_ok=True)

    print("Ratios:")
    print(
        f"  nonb_ratio = {args.nonb_ratio} "
        "= P(true non-B | labeled non-B)"
    )
    print(
        f"  b_ratio    = {args.b_ratio} "
        "= P(true non-B | labeled B-DNA)"
    )
    print("\nPaper splits:")
    print("  labeled B-DNA: 30% train / 20% val / 50% test")
    print("  labeled non-B: 10% train / 10% val / 80% test\n")

    for offset, nonb_type in enumerate(nonb_types):
        simulate_one_type(
            outdir=args.output,
            nonb_type=nonb_type,
            n_nonb=args.n_nonb,
            n_b=args.n_b,
            nonb_ratio=args.nonb_ratio,
            b_ratio=args.b_ratio,
            window_size=args.window_size,
            noise_sd=args.noise_sd,
            seed=args.seed + offset,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
    