"""
download_gtsrb.py
-----------------
Downloads the GTSRB dataset and organizes it into:
  data/processed/
    train/  (class subfolders)
    val/    (class subfolders, stratified 80/20 split)
    test/   (class subfolders)

Usage:
  python data/download_gtsrb.py
"""

import os
import shutil
import random
import csv
from pathlib import Path
from collections import defaultdict


# ── Constants ─────────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw/GTSRB")
PROCESSED_DIR = Path("data/processed")
TRAIN_DIR     = PROCESSED_DIR / "train"
VAL_DIR       = PROCESSED_DIR / "val"
TEST_DIR      = PROCESSED_DIR / "test"
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42

# Human-readable names for all 43 classes
CLASS_NAMES = {
    0:  "Speed limit 20",
    1:  "Speed limit 30",
    2:  "Speed limit 50",
    3:  "Speed limit 60",
    4:  "Speed limit 70",
    5:  "Speed limit 80",
    6:  "End of speed limit 80",
    7:  "Speed limit 100",
    8:  "Speed limit 120",
    9:  "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End all speed/passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing for 3.5t",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_kaggle():
    """Verify kaggle is installed and credentials exist."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError(
            "kaggle package not found. Run: pip install kaggle\n"
            "Then place your kaggle.json at ~/.kaggle/kaggle.json"
        )
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {creds}.\n"
            "Download kaggle.json from https://www.kaggle.com/settings → API"
        )


def download_dataset():
    """Download GTSRB from Kaggle."""
    import kaggle
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading GTSRB dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        path=str(raw_dir),
        unzip=True,
    )
    print(f"✅ Downloaded to {raw_dir}")


def find_raw_train_dir() -> Path:
    """Locate the Train/ folder regardless of Kaggle extraction layout."""
    candidates = [
        RAW_DIR / "Train",
        Path("data/raw/Train"),
        Path("data/raw/GTSRB/Final_Training/Images"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find the raw Train directory. "
        "Check your extraction path and update RAW_DIR."
    )


def find_raw_test_dir() -> Path:
    """Locate the Test/ folder."""
    candidates = [
        RAW_DIR / "Test",
        Path("data/raw/Test"),
        Path("data/raw/GTSRB/Final_Test/Images"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find the raw Test directory.")


def find_test_csv() -> Path | None:
    """Locate GT-final_test.csv that maps test images to labels."""
    candidates = [
        Path("data/raw/Test.csv"),
        Path("data/raw/GTSRB/GT-final_test.csv"),
        RAW_DIR / "GT-final_test.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def split_train_val(train_src: Path):
    """
    Copy images from raw Train/ into processed train/ and val/,
    using a stratified 80/20 split per class.
    """
    random.seed(RANDOM_SEED)
    total_train, total_val = 0, 0
    class_counts = defaultdict(lambda: {"train": 0, "val": 0})

    class_dirs = sorted(train_src.iterdir())
    print(f"\n📂 Splitting {len(class_dirs)} classes into train/val...")

    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue

        label = int(class_dir.name)
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.ppm"))

        if not images:
            print(f"  ⚠️  No images found in {class_dir}, skipping.")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_SPLIT))
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest_dir = (TRAIN_DIR if split == "train" else VAL_DIR) / str(label)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest_dir / img.name)
            class_counts[label][split] = len(imgs)

        total_train += len(train_imgs)
        total_val   += len(val_imgs)

    print(f"  ✅ Train: {total_train} images | Val: {total_val} images")
    return class_counts


def detect_csv_columns(csv_path: Path) -> tuple[str, str, str]:
    """
    Sniff the actual column names for filename, class, and delimiter.
    Returns (filename_col, classid_col, delimiter).
    """
    # Try semicolon first, then comma
    for delim in (";", ","):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            headers = reader.fieldnames or []
            headers_lower = [h.strip().lower() for h in headers]

        # Match filename column
        fname_col = next(
            (headers[i] for i, h in enumerate(headers_lower)
             if h in ("filename", "path", "file")), None
        )
        # Match class column
        class_col = next(
            (headers[i] for i, h in enumerate(headers_lower)
             if h in ("classid", "class_id", "class", "label", "category")), None
        )

        if fname_col and class_col:
            print(f"  📋 CSV columns detected — file: '{fname_col}', class: '{class_col}', delimiter: '{delim}'")
            return fname_col, class_col, delim

    # Print headers to help debug
    with open(csv_path, newline="") as f:
        first_lines = [f.readline() for _ in range(3)]
    print("  ⚠️  Could not auto-detect CSV columns. First 3 lines:")
    for line in first_lines:
        print(f"      {line.rstrip()}")
    raise ValueError(
        "Cannot detect filename/classid columns in CSV. "
        "Check the printed headers above and update detect_csv_columns()."
    )


def organize_test(test_src: Path, csv_path: Path | None):
    """
    Copy test images into processed test/<class>/ subfolders.
    Uses the CSV label file if available.
    """
    print("\n📂 Organizing test set...")
    copied = 0

    if csv_path:
        fname_col, class_col, delim = detect_csv_columns(csv_path)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                fname = Path(row[fname_col].strip()).name
                label = int(row[class_col].strip())

                # Try direct match, then subfolder match
                src = test_src / fname
                if not src.exists():
                    src = test_src / row[fname_col].strip()
                if not src.exists():
                    # Some Kaggle versions nest test images in a subfolder
                    matches = list(test_src.rglob(fname))
                    src = matches[0] if matches else src

                if not src.exists():
                    continue

                dest_dir = TEST_DIR / str(label)
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_dir / fname)
                copied += 1
    else:
        # Fallback: copy flat test images without labels (for inference only)
        misc_dir = TEST_DIR / "unlabeled"
        misc_dir.mkdir(parents=True, exist_ok=True)
        for img in test_src.rglob("*.ppm"):
            shutil.copy2(img, misc_dir / img.name)
            copied += 1

    print(f"  ✅ Test: {copied} images organized")


def save_class_names():
    """Write class_names.txt for reference."""
    out = PROCESSED_DIR / "class_names.txt"
    with open(out, "w") as f:
        for idx, name in CLASS_NAMES.items():
            f.write(f"{idx:02d}: {name}\n")
    print(f"\n📝 Class names saved to {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  GTSRB Dataset Download & Organisation")
    print("=" * 60)

    # 1. Download if raw data doesn't exist
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        check_kaggle()
        download_dataset()
    else:
        print(f"✅ Raw data already exists at {RAW_DIR}, skipping download.")

    # 2. Split train → train/val
    train_src = find_raw_train_dir()
    class_counts = split_train_val(train_src)

    # 3. Organize test set
    test_src  = find_raw_test_dir()
    csv_path  = find_test_csv()
    organize_test(test_src, csv_path)

    # 4. Save class name mapping
    save_class_names()

    # 5. Summary
    print("\n" + "=" * 60)
    print("  ✅ Dataset ready!")
    print(f"  Train : {TRAIN_DIR}")
    print(f"  Val   : {VAL_DIR}")
    print(f"  Test  : {TEST_DIR}")
    print("=" * 60)

    # Print per-class distribution
    print("\nPer-class split summary:")
    print(f"  {'Class':<6} {'Name':<40} {'Train':>6} {'Val':>5}")
    print("  " + "-" * 60)
    for label in sorted(class_counts):
        name = CLASS_NAMES.get(label, "Unknown")
        tr   = class_counts[label]["train"]
        vl   = class_counts[label]["val"]
        print(f"  {label:<6} {name:<40} {tr:>6} {vl:>5}")


if __name__ == "__main__":
    main()