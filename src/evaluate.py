"""
evaluate.py
-----------
Runs the trained model against the held-out GTSRB test set and produces:

  results/confusion_matrix.png   — 43×43 heatmap
  results/class_report.txt       — per-class precision / recall / F1
  results/metrics.json           — updated with test metrics
  results/worst_classes.png      — bar chart of the 10 lowest F1 classes

Usage:
    python -m src.evaluate
    python -m src.evaluate --weights weights/best_model.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

from src.augmentation import load_config
from src.dataset import GTSRBDataset, build_dataloaders
from src.models.classifier import get_model, get_device


# ── Class names (short labels for confusion matrix axis) ──────────────────────

SHORT_NAMES = {
    0:  "Spd 20",    1:  "Spd 30",    2:  "Spd 50",
    3:  "Spd 60",    4:  "Spd 70",    5:  "Spd 80",
    6:  "End S80",   7:  "Spd 100",   8:  "Spd 120",
    9:  "No pass",   10: "NP 3.5t",   11: "Right-of-way",
    12: "Priority",  13: "Yield",     14: "Stop",
    15: "No veh",    16: "NV 3.5t",   17: "No entry",
    18: "Caution",   19: "Curve L",   20: "Curve R",
    21: "Dbl crv",   22: "Bumpy",     23: "Slippery",
    24: "Narrows",   25: "Roadwork",  26: "Signals",
    27: "Pedestrian",28: "Children",  29: "Bicycles",
    30: "Ice/snow",  31: "Animals",   32: "End lim",
    33: "Turn R",    34: "Turn L",    35: "Ahead",
    36: "Str/R",     37: "Str/L",     38: "Keep R",
    39: "Keep L",    40: "Roundabout",41: "End NP",
    42: "End NP3.5",
}

FULL_NAMES = {
    0:  "Speed limit 20",        1:  "Speed limit 30",
    2:  "Speed limit 50",        3:  "Speed limit 60",
    4:  "Speed limit 70",        5:  "Speed limit 80",
    6:  "End of speed limit 80", 7:  "Speed limit 100",
    8:  "Speed limit 120",       9:  "No passing",
    10: "No passing (3.5t)",     11: "Right-of-way",
    12: "Priority road",         13: "Yield",
    14: "Stop",                  15: "No vehicles",
    16: "No vehicles (3.5t)",    17: "No entry",
    18: "General caution",       19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road",            23: "Slippery road",
    24: "Road narrows right",    25: "Road work",
    26: "Traffic signals",       27: "Pedestrians",
    28: "Children crossing",     29: "Bicycles crossing",
    30: "Beware ice/snow",       31: "Wild animals crossing",
    32: "End all limits",        33: "Turn right ahead",
    34: "Turn left ahead",       35: "Ahead only",
    36: "Go straight or right",  37: "Go straight or left",
    38: "Keep right",            39: "Keep left",
    40: "Roundabout mandatory",  41: "End of no passing",
    42: "End no passing (3.5t)",
}


# ── Inference on test set ──────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (all_preds, all_labels) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Evaluating", ncols=80):
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
):
    n = 43
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    # Normalise row-wise (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    labels = [SHORT_NAMES[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Recall (row-normalised)"},
    )
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label",      fontsize=13)
    ax.set_title(
        "GTSRB — Confusion Matrix (row-normalised)\n"
        "Diagonal = correct; off-diagonal = errors",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# ── Per-class accuracy bar chart ───────────────────────────────────────────────

def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
):
    n = 43
    accs = []
    for cls in range(n):
        mask = y_true == cls
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((y_pred[mask] == cls).mean())

    labels  = [SHORT_NAMES[i] for i in range(n)]
    colors  = ["#2ecc71" if a >= 0.95 else
               "#f39c12" if a >= 0.85 else
               "#e74c3c" for a in accs]

    fig, ax = plt.subplots(figsize=(20, 6))
    bars = ax.bar(range(n), [a * 100 for a in accs], color=colors, edgecolor="white")
    ax.axhline(95, color="gray", linestyle="--", linewidth=1.2, label="95% threshold")
    ax.axhline(np.mean(accs) * 100, color="navy", linestyle="-.",
               linewidth=1.5, label=f"Mean {np.mean(accs)*100:.1f}%")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("Per-Class Test Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Colour legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#2ecc71", label="≥ 95%"),
        Patch(facecolor="#f39c12", label="85–95%"),
        Patch(facecolor="#e74c3c", label="< 85%"),
    ]
    ax.legend(handles=legend_els + ax.get_legend_handles_labels()[0][:2],
              fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")
    return accs


# ── Worst classes ──────────────────────────────────────────────────────────────

def plot_worst_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    n_worst: int = 10,
):
    report = classification_report(
        y_true, y_pred,
        labels=list(range(43)),
        output_dict=True,
        zero_division=0,
    )
    f1_scores = {
        int(k): v["f1-score"]
        for k, v in report.items()
        if k.isdigit()
    }
    worst = sorted(f1_scores.items(), key=lambda x: x[1])[:n_worst]
    classes, scores = zip(*worst)
    names  = [FULL_NAMES[c] for c in classes]
    colors = ["#e74c3c" if s < 0.85 else "#f39c12" for s in scores]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(range(n_worst), [s * 100 for s in scores],
                   color=colors, edgecolor="white")
    ax.set_yticks(range(n_worst))
    ax.set_yticklabels([f"[{c:02d}] {n}" for c, n in zip(classes, names)], fontsize=10)
    ax.set_xlabel("F1 Score (%)", fontsize=12)
    ax.set_title(f"Top-{n_worst} Hardest Classes (lowest F1)", fontsize=13, fontweight="bold")
    ax.axvline(95, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score*100:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# ── Save class report ──────────────────────────────────────────────────────────

def save_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
):
    names = [FULL_NAMES[i] for i in range(43)]
    report = classification_report(
        y_true, y_pred,
        target_names=names,
        digits=4,
        zero_division=0,
    )
    with open(save_path, "w") as f:
        f.write("GTSRB Test Set — Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"  ✅ Saved: {save_path}")
    return report


# ── Update metrics.json ────────────────────────────────────────────────────────

def update_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_path: Path,
):
    test_metrics = {
        "test_accuracy":    round(accuracy_score(y_true, y_pred), 6),
        "test_macro_f1":    round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 6),
        "test_weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 6),
    }

    existing = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    existing.update(test_metrics)

    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    return test_metrics


# ── Print summary ──────────────────────────────────────────────────────────────

def print_summary(test_metrics: dict, accs: list, y_true, y_pred):
    n_correct = (np.array(y_true) == np.array(y_pred)).sum()
    n_total   = len(y_true)
    below_95  = sum(1 for a in accs if a < 0.95)

    print("\n" + "=" * 55)
    print("  GTSRB Test Set Results")
    print("=" * 55)
    print(f"  Test accuracy    : {test_metrics['test_accuracy']*100:.3f}%")
    print(f"  Macro F1         : {test_metrics['test_macro_f1']*100:.3f}%")
    print(f"  Weighted F1      : {test_metrics['test_weighted_f1']*100:.3f}%")
    print(f"  Correct / Total  : {n_correct:,} / {n_total:,}")
    print(f"  Classes < 95%    : {below_95} / 43")
    print("=" * 55)


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(
    weights_path: str = "weights/best_model.pth",
    config_path:  str = "configs/train_config.yaml",
):
    config      = load_config(config_path)
    device      = get_device(config)
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("  Traffic Sign Classifier — Evaluation")
    print("=" * 55)
    print(f"  Weights : {weights_path}")
    print(f"  Device  : {device}\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt = torch.load(weights_path, map_location=device)
    model = get_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_acc={ckpt['val_acc']:.4f})\n")

    # ── Test loader ────────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(config_path)

    # ── Run inference ──────────────────────────────────────────────────────────
    y_pred, y_true = run_inference(model, test_loader, device)

    # ── Generate outputs ───────────────────────────────────────────────────────
    print("\n📊 Generating evaluation outputs...")

    plot_confusion_matrix(
        y_true, y_pred,
        results_dir / "confusion_matrix.png",
    )
    accs = plot_per_class_accuracy(
        y_true, y_pred,
        results_dir / "per_class_accuracy.png",
    )
    plot_worst_classes(
        y_true, y_pred,
        results_dir / "worst_classes.png",
    )
    report = save_class_report(
        y_true, y_pred,
        results_dir / "class_report.txt",
    )
    test_metrics = update_metrics(
        y_true, y_pred,
        Path(config["paths"]["metrics"]),
    )

    print_summary(test_metrics, accs, y_true, y_pred)
    print("\n✅ Evaluation complete!\n")

    return y_true, y_pred, test_metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/best_model.pth")
    parser.add_argument("--config",  default="configs/train_config.yaml")
    args = parser.parse_args()
    evaluate(args.weights, args.config)