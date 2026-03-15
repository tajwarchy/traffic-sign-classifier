"""
data_stats.py
-------------
Computes and visualises dataset statistics for GTSRB.

Outputs:
  results/class_distribution.png  — bar chart of per-class counts
  results/image_size_dist.png     — histogram of image dimensions
  results/pixel_stats.txt         — channel mean/std for normalisation

Usage:
  python data/data_stats.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_DIR   = Path("data/processed/train")
VAL_DIR     = Path("data/processed/val")
TEST_DIR    = Path("data/processed/test")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Speed 20",    1: "Speed 30",    2: "Speed 50",
    3: "Speed 60",    4: "Speed 70",    5: "Speed 80",
    6: "End Spd 80",  7: "Speed 100",   8: "Speed 120",
    9: "No pass",    10: "No pass 3.5t",11: "Right-of-way",
   12: "Priority",   13: "Yield",       14: "Stop",
   15: "No vehicle", 16: "No veh 3.5t", 17: "No entry",
   18: "Caution",    19: "Curve L",     20: "Curve R",
   21: "Dbl curve",  22: "Bumpy",       23: "Slippery",
   24: "Narrows R",  25: "Road work",   26: "Traffic sig",
   27: "Pedestrian", 28: "Children",    29: "Bicycles",
   30: "Ice/snow",   31: "Wild animal", 32: "End limits",
   33: "Turn R",     34: "Turn L",      35: "Ahead only",
   36: "Str or R",   37: "Str or L",    38: "Keep R",
   39: "Keep L",     40: "Roundabout",  41: "End no pass",
   42: "End no p 3.5t",
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def count_images(split_dir: Path) -> dict[int, int]:
    """Return {class_id: count} for a given split directory."""
    counts = {}
    if not split_dir.exists():
        return counts
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir() and class_dir.name.isdigit():
            imgs = list(class_dir.glob("*.png")) + \
                   list(class_dir.glob("*.ppm")) + \
                   list(class_dir.glob("*.jpg"))
            counts[int(class_dir.name)] = len(imgs)
    return counts


def gather_all_images(split_dir: Path) -> list[Path]:
    imgs = []
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            imgs += list(class_dir.glob("*.png"))
            imgs += list(class_dir.glob("*.ppm"))
            imgs += list(class_dir.glob("*.jpg"))
    return imgs


# ── Plot 1: Class Distribution ─────────────────────────────────────────────────

def plot_class_distribution(train_counts: dict, val_counts: dict):
    labels   = list(range(43))
    tr_vals  = [train_counts.get(i, 0) for i in labels]
    vl_vals  = [val_counts.get(i, 0)   for i in labels]
    names    = [CLASS_NAMES.get(i, str(i)) for i in labels]

    x = np.arange(len(labels))
    w = 0.6

    fig, ax = plt.subplots(figsize=(20, 7))
    bars = ax.bar(x, tr_vals, width=w, label="Train", color="#4C72B0", alpha=0.85)
    ax.bar(x, vl_vals, width=w, bottom=tr_vals, label="Val", color="#DD8452", alpha=0.85)

    # Annotate min/max
    max_cls = max(train_counts, key=train_counts.get)
    min_cls = min(train_counts, key=train_counts.get)
    ax.annotate(f"Max: {tr_vals[max_cls]}",
                xy=(max_cls, tr_vals[max_cls]), xytext=(max_cls, tr_vals[max_cls] + 80),
                ha="center", fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))
    ax.annotate(f"Min: {tr_vals[min_cls]}",
                xy=(min_cls, tr_vals[min_cls]), xytext=(min_cls, tr_vals[min_cls] + 80),
                ha="center", fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    ax.set_title("GTSRB Class Distribution (Train + Val)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = RESULTS_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {out}")
    return tr_vals


# ── Plot 2: Image Size Distribution ───────────────────────────────────────────

def plot_image_size_distribution(images: list[Path], sample_n: int = 2000):
    """Sample up to sample_n images and plot width/height histograms."""
    rng = np.random.default_rng(42)
    sampled = rng.choice(images, size=min(sample_n, len(images)), replace=False)

    widths, heights = [], []
    for p in tqdm(sampled, desc="  Reading image sizes", leave=False):
        try:
            w, h = Image.open(p).size
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, label, color in zip(
        axes,
        [widths, heights],
        ["Width (px)", "Height (px)"],
        ["#4C72B0", "#DD8452"],
    ):
        ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(np.median(vals), color="black", linestyle="--",
                   label=f"Median: {np.median(vals):.0f}px")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Image {label} Distribution", fontsize=12)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("GTSRB Image Size Distribution (sampled)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "image_size_dist.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {out}")

    return {
        "width_min": int(min(widths)),  "width_max": int(max(widths)),
        "width_median": float(np.median(widths)),
        "height_min": int(min(heights)), "height_max": int(max(heights)),
        "height_median": float(np.median(heights)),
    }


# ── Compute Channel Mean/Std ───────────────────────────────────────────────────

def compute_channel_stats(images: list[Path], sample_n: int = 3000) -> dict:
    """Compute per-channel mean and std over a random sample of training images."""
    rng = np.random.default_rng(42)
    sampled = rng.choice(images, size=min(sample_n, len(images)), replace=False)

    means, stds = [], []
    for p in tqdm(sampled, desc="  Computing channel stats", leave=False):
        try:
            img = np.array(Image.open(p).convert("RGB").resize((64, 64))).astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
        except Exception:
            continue

    mean = np.mean(means, axis=0).tolist()
    std  = np.mean(stds,  axis=0).tolist()
    return {"mean": [round(m, 4) for m in mean],
            "std":  [round(s, 4) for s in std]}


# ── Print Summary Table ────────────────────────────────────────────────────────

def print_summary(train_counts, val_counts, test_counts, tr_vals):
    total_tr  = sum(train_counts.values())
    total_vl  = sum(val_counts.values())
    total_te  = sum(test_counts.values())
    max_cls   = max(train_counts, key=train_counts.get)
    min_cls   = min(train_counts, key=train_counts.get)
    imbalance = train_counts[max_cls] / max(train_counts[min_cls], 1)

    print("\n" + "=" * 55)
    print("  GTSRB Dataset Statistics")
    print("=" * 55)
    print(f"  Classes            : 43")
    print(f"  Train images       : {total_tr:,}")
    print(f"  Val images         : {total_vl:,}")
    print(f"  Test images        : {total_te:,}")
    print(f"  Total              : {total_tr + total_vl + total_te:,}")
    print(f"  Most common class  : {max_cls} ({CLASS_NAMES[max_cls]}) — {train_counts[max_cls]:,} imgs")
    print(f"  Rarest class       : {min_cls} ({CLASS_NAMES[min_cls]}) — {train_counts[min_cls]:,} imgs")
    print(f"  Imbalance ratio    : {imbalance:.1f}x")
    print("=" * 55)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n🔍 Computing GTSRB dataset statistics...\n")

    train_counts = count_images(TRAIN_DIR)
    val_counts   = count_images(VAL_DIR)
    test_counts  = count_images(TEST_DIR)

    if not train_counts:
        print("❌ No training data found. Run download_gtsrb.py first.")
        return

    # Plot class distribution
    print("📊 Plotting class distribution...")
    tr_vals = plot_class_distribution(train_counts, val_counts)

    # Image size distribution
    print("📐 Analysing image sizes...")
    all_train_imgs = gather_all_images(TRAIN_DIR)
    size_stats     = plot_image_size_distribution(all_train_imgs)

    # Channel stats (for normalisation)
    print("🎨 Computing channel mean/std...")
    ch_stats = compute_channel_stats(all_train_imgs)
    print(f"  Channel mean : {ch_stats['mean']}")
    print(f"  Channel std  : {ch_stats['std']}")

    # Save pixel stats to file
    stats = {
        "channel_stats": ch_stats,
        "image_size_stats": size_stats,
        "class_counts": {
            "train": train_counts,
            "val":   val_counts,
            "test":  test_counts,
        },
    }
    stats_path = RESULTS_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ Full stats saved: {stats_path}")

    # Save normalisation values to txt
    pix_path = RESULTS_DIR / "pixel_stats.txt"
    with open(pix_path, "w") as f:
        f.write(f"mean: {ch_stats['mean']}\n")
        f.write(f"std:  {ch_stats['std']}\n")
    print(f"  ✅ Pixel stats saved: {pix_path}")

    print_summary(train_counts, val_counts, test_counts, tr_vals)
    print("\n✅ All statistics computed successfully!\n")


if __name__ == "__main__":
    main()