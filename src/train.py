"""
train.py
--------
Two-phase training loop for TrafficSignClassifier on GTSRB.

Phase 1 (epochs 1 → freeze_epochs):
    Backbone frozen. Only the classifier head is trained.
    High LR is safe here — no risk of destroying pretrained weights.

Phase 2 (epochs freeze_epochs+1 → total_epochs):
    Full fine-tuning. Backbone unfrozen with a lower LR.
    CosineAnnealingLR brings LR smoothly to near-zero by the last epoch.

Usage:
    python -m src.train
    python -m src.train --config configs/train_config.yaml
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from src.augmentation import load_config
from src.dataset import build_dataloaders
from src.models.classifier import get_model, get_device


# ── Helpers ────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    """
    Separate param groups so backbone and head can have different LRs.
    During Phase 1 backbone params have requires_grad=False so they
    simply won't receive any gradient updates.
    """
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for p in model.head.parameters()     if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * 0.1})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})

    return AdamW(param_groups, weight_decay=weight_decay)


def get_scheduler(name: str, optimizer, epochs: int, steps_per_epoch: int):
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,
        )
    raise ValueError(f"Unknown scheduler: {name}")


# ── One epoch ──────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    is_train: bool,
    use_onecycle: bool = False,
) -> tuple[float, float]:
    """
    Runs one full pass over the loader.
    Returns (avg_loss, avg_accuracy).
    """
    model.train() if is_train else model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "  Train" if is_train else "  Val  "

    with ctx:
        for imgs, labels in tqdm(loader, desc=desc, leave=False, ncols=80):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                # Gradient clipping — prevents rare exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if use_onecycle:
                    scheduler.step()   # OneCycleLR steps per batch

            total_loss += loss.item()
            total_acc  += accuracy(logits, labels)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── Main training loop ─────────────────────────────────────────────────────────

def train(config_path: str = "configs/train_config.yaml"):
    config     = load_config(config_path)
    train_cfg  = config["training"]
    model_cfg  = config["model"]
    paths      = config["paths"]

    device        = get_device(config)
    epochs        = train_cfg["epochs"]
    freeze_epochs = model_cfg["freeze_epochs"]
    lr            = train_cfg["learning_rate"]
    wd            = train_cfg["weight_decay"]
    scheduler_name= train_cfg["scheduler"]
    weights_dir   = Path(paths["weights_dir"])
    results_dir   = Path(paths["results_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Traffic Sign Classifier — Training")
    print("=" * 60)
    print(f"  Device        : {device}")
    print(f"  Epochs        : {epochs}  (freeze for first {freeze_epochs})")
    print(f"  Batch size    : {train_cfg['batch_size']}")
    print(f"  LR            : {lr}  |  WD: {wd}")
    print(f"  Scheduler     : {scheduler_name}")
    print("=" * 60 + "\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    print("📂 Loading datasets...")
    train_loader, val_loader, _ = build_dataloaders(config_path)
    train_ds = train_loader.dataset

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n🏗️  Building model...")
    model = get_model(config).to(device)
    model.set_phase(1)           # start with frozen backbone
    model.param_summary()

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=train_ds.get_class_weights().to(device)
        if train_cfg["use_weighted_loss"] else None
    )
    print(f"\n  Loss: CrossEntropyLoss  "
          f"(weighted={train_cfg['use_weighted_loss']})")

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    # Phase 1: only head params; we rebuild optimizer at phase transition
    optimizer = get_optimizer(model, lr, wd)
    use_onecycle = (scheduler_name == "onecycle")
    scheduler = get_scheduler(
        scheduler_name, optimizer,
        epochs=epochs - freeze_epochs,    # cosine over Phase-2 only
        steps_per_epoch=len(train_loader),
    )

    # ── State ──────────────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_epoch     = 0
    history        = []
    start_time     = time.time()
    phase_switched = False

    print(f"\n🚀 Starting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Phase transition ───────────────────────────────────────────────────
        if epoch == freeze_epochs + 1 and not phase_switched:
            print(f"\n{'─'*60}")
            print(f"  🔓 Epoch {epoch}: Switching to Phase 2 — full fine-tuning")
            print(f"{'─'*60}\n")
            model.set_phase(2)
            model.param_summary()

            # Rebuild optimizer: backbone gets LR/10, head gets LR/3
            optimizer = get_optimizer(model, lr / 3, wd)
            scheduler = get_scheduler(
                scheduler_name, optimizer,
                epochs=epochs - freeze_epochs,
                steps_per_epoch=len(train_loader),
            )
            phase_switched = True

        # ── Train ──────────────────────────────────────────────────────────────
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, device, is_train=True,
            use_onecycle=use_onecycle,
        )

        # ── Validate ───────────────────────────────────────────────────────────
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer,
            scheduler, device, is_train=False,
        )

        # CosineAnnealingLR steps per epoch (not per batch)
        if not use_onecycle and phase_switched:
            scheduler.step()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "config":     config,
            }, weights_dir / "best_model.pth")
            saved_tag = "  ✅ saved"
        else:
            saved_tag = ""

        # Current LR (first param group)
        cur_lr = optimizer.param_groups[-1]["lr"]

        # ── Log ───────────────────────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        phase   = 1 if epoch <= freeze_epochs else 2
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"[P{phase}]  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={cur_lr:.2e}  "
            f"({elapsed:.1f}s){saved_tag}"
        )

        history.append({
            "epoch":      epoch,
            "phase":      phase,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
            "lr":         cur_lr,
        })

    # ── Final summary ──────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ✅ Training complete in {format_time(total_time)}")
    print(f"  Best val accuracy : {best_val_acc:.4f} at epoch {best_epoch}")
    print("=" * 60 + "\n")

    # Save training history
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "best_epoch":   best_epoch,
            "total_time_s": round(total_time, 1),
            "history":      history,
        }, f, indent=2)
    print(f"📊 Metrics saved → {metrics_path}")
    print(f"💾 Best model saved → {weights_dir / 'best_model.pth'}\n")

    return model, history


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to train_config.yaml",
    )
    args = parser.parse_args()
    train(args.config)