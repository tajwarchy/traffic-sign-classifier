"""
augmentation.py
---------------
Albumentations-based augmentation pipelines for GTSRB.

All parameters are driven by configs/train_config.yaml so
nothing needs to be changed here when experimenting.
"""

import yaml
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_train_transforms(config: dict) -> A.Compose:
    """
    Heavy augmentation pipeline for training.
    Simulates real-world conditions: lighting changes, motion blur,
    noise, slight rotations, and occlusion (coarse dropout).
    """
    aug  = config["augmentation"]["train"]
    norm = config["augmentation"]["normalize"]
    size = config["training"]["image_size"]

    transforms = [A.Resize(size, size)]

    if aug.get("random_brightness_contrast"):
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_limit"],
            contrast_limit=aug["contrast_limit"],
            p=0.6,
        ))

    if aug.get("clahe"):
        transforms.append(A.CLAHE(clip_limit=4.0, p=0.4))

    if aug.get("motion_blur"):
        transforms.append(A.MotionBlur(
            blur_limit=aug["blur_limit"],
            p=0.3,
        ))

    if aug.get("gauss_noise"):
        transforms.append(A.GaussNoise(p=0.3))

    if aug.get("shift_scale_rotate"):
        transforms.append(A.ShiftScaleRotate(
            shift_limit=aug["shift_limit"],
            scale_limit=aug["scale_limit"],
            rotate_limit=aug["rotate_limit"],
            border_mode=0,
            p=0.6,
        ))

    # Perspective distortion — simulates camera angle variation
    transforms.append(A.Perspective(scale=(0.02, 0.08), p=0.3))

    # Simulate shadows across the sign
    transforms.append(A.RandomShadow(p=0.2))

    if aug.get("coarse_dropout"):
        transforms.append(A.CoarseDropout(
            max_holes=aug["max_holes"],
            max_height=aug["max_height"],
            max_width=aug["max_width"],
            fill_value=0,
            p=0.3,
        ))

    transforms += [
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(config: dict) -> A.Compose:
    """
    Minimal pipeline for validation and testing.
    Only resize + normalize — no randomness.
    """
    norm = config["augmentation"]["normalize"]
    size = config["training"]["image_size"]

    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


def get_inference_transforms(config: dict) -> A.Compose:
    """Same as val transforms — used during inference on single images."""
    return get_val_transforms(config)


# ── Visualisation helper ───────────────────────────────────────────────────────

def visualize_augmentations(
    image_path: str,
    config_path: str = "configs/train_config.yaml",
    n_samples: int = 8,
    save_path: str = "results/augmentation_preview.png",
):
    """
    Show original + N augmented versions of a single image side by side.
    Useful for verifying augmentation strength before training.

    Usage (in notebook):
        from src.augmentation import visualize_augmentations
        visualize_augmentations("data/processed/train/14/00014_00000_00000.png")
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    config    = load_config(config_path)
    transform = get_train_transforms(config)

    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)

    fig, axes = plt.subplots(2, (n_samples + 2) // 2, figsize=(18, 6))
    axes = axes.flatten()

    # Original (no transform — just resize)
    resized = A.Resize(config["training"]["image_size"],
                       config["training"]["image_size"])(image=img_np)["image"]
    axes[0].imshow(resized)
    axes[0].set_title("Original", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i in range(1, n_samples + 1):
        aug_img = transform(image=img_np)["image"]
        # Denormalise for display: move channel dim and undo normalisation
        mean = np.array(config["augmentation"]["normalize"]["mean"])
        std  = np.array(config["augmentation"]["normalize"]["std"])
        disp = aug_img.permute(1, 2, 0).numpy()
        disp = (disp * std + mean).clip(0, 1)
        axes[i].imshow(disp)
        axes[i].set_title(f"Aug {i}", fontsize=9)
        axes[i].axis("off")

    # Hide any unused axes
    for j in range(n_samples + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Augmentation Preview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved augmentation preview → {save_path}")