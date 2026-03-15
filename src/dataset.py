"""
dataset.py
----------
GTSRBDataset — PyTorch Dataset for the organised GTSRB folder structure.

Expected layout (produced by download_gtsrb.py):
  data/processed/
    train/  0/  1/  ... 42/
    val/    0/  1/  ... 42/
    test/   0/  1/  ... 42/

Key features:
  - Reads images as RGB numpy arrays (Albumentations-compatible)
  - Returns class weights for weighted CrossEntropyLoss
  - Returns sample weights for optional WeightedRandomSampler
  - Clean label-to-name mapping baked in
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from src.augmentation import (
    get_train_transforms,
    get_val_transforms,
    get_inference_transforms,
    load_config,
)


# ── Class name mapping ─────────────────────────────────────────────────────────

CLASS_NAMES = {
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

IMG_EXTENSIONS = {".png", ".ppm", ".jpg", ".jpeg"}


# ── Dataset ────────────────────────────────────────────────────────────────────

class GTSRBDataset(Dataset):
    """
    Parameters
    ----------
    root_dir  : Path to a split folder (train/, val/, or test/).
    transform : Albumentations Compose pipeline.
    """

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []  # (image_path, label)

        self._load_samples()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_samples(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            label = int(class_dir.name)
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root_dir}. "
                               "Run download_gtsrb.py first.")

        print(f"  Loaded {len(self.samples):,} samples from {self.root_dir}")

    # ── Public helpers ─────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return len({label for _, label in self.samples})

    @property
    def class_counts(self) -> dict[int, int]:
        return dict(Counter(label for _, label in self.samples))

    def get_class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for CrossEntropyLoss(weight=...).
        Rare classes get higher weight → model penalised more for missing them.
        """
        counts = self.class_counts
        n_classes = max(counts.keys()) + 1
        weights = torch.zeros(n_classes)
        total = sum(counts.values())
        for cls, cnt in counts.items():
            weights[cls] = total / (n_classes * cnt)
        # Normalise so mean weight ≈ 1
        weights = weights / weights.mean()
        return weights

    def get_sample_weights(self) -> list[float]:
        """
        Per-sample weights for WeightedRandomSampler.
        Each sample gets the inverse frequency of its class.
        """
        counts = self.class_counts
        total  = sum(counts.values())
        n_cls  = max(counts.keys()) + 1
        cls_w  = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
        return [cls_w[label] for _, label in self.samples]

    def label_to_name(self, label: int) -> str:
        return CLASS_NAMES.get(label, f"Class {label}")

    # ── PyTorch interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    config_path: str = "configs/train_config.yaml",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train, val, and test DataLoaders from config.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    config = load_config(config_path)
    data   = config["data"]
    train_cfg = config["training"]

    train_tf = get_train_transforms(config)
    val_tf   = get_val_transforms(config)

    train_ds = GTSRBDataset(data["train_dir"], transform=train_tf)
    val_ds   = GTSRBDataset(data["val_dir"],   transform=val_tf)
    test_ds  = GTSRBDataset(data["test_dir"],  transform=val_tf)

    # Optional: WeightedRandomSampler to over-sample rare classes
    sampler = None
    if train_cfg.get("use_weighted_sampler"):
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print("  Using WeightedRandomSampler for training.")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),   # shuffle only if no sampler
        sampler=sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=train_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=train_cfg["num_workers"] > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    return train_loader, val_loader, test_loader


# ── Sanity check ───────────────────────────────────────────────────────────────

def dataset_sanity_check(config_path: str = "configs/train_config.yaml"):
    """
    Quick sanity check — prints shapes, label sample, and class weight stats.
    Run directly: python -c "from src.dataset import dataset_sanity_check; dataset_sanity_check()"
    """
    print("\n🔍 Running dataset sanity check...\n")
    train_loader, val_loader, test_loader = build_dataloaders(config_path)

    imgs, labels = next(iter(train_loader))
    print(f"  Batch image shape : {imgs.shape}")        # (B, 3, 224, 224)
    print(f"  Batch label shape : {labels.shape}")      # (B,)
    print(f"  Label sample      : {labels[:8].tolist()}")
    print(f"  Image dtype       : {imgs.dtype}")
    print(f"  Image min/max     : {imgs.min():.3f} / {imgs.max():.3f}")

    train_ds = train_loader.dataset
    weights  = train_ds.get_class_weights()
    print(f"\n  Class weights (first 10): {weights[:10].tolist()}")
    print(f"  Weight min: {weights.min():.3f}  max: {weights.max():.3f}  mean: {weights.mean():.3f}")

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")
    print("\n✅ Sanity check passed!\n")


if __name__ == "__main__":
    dataset_sanity_check()