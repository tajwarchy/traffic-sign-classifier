"""
inference.py
------------
Three modes:

  1. Single image
       python -m src.inference --image path/to/sign.jpg

  2. Batch folder
       python -m src.inference --folder path/to/folder --output results/predictions.csv

  3. Real-time webcam / video
       python -m src.inference --webcam
       python -m src.inference --video path/to/video.mp4

Controls (webcam/video):
  Q  — quit
  S  — save screenshot to results/screenshots/
  +  — increase confidence threshold
  -  — decrease confidence threshold
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.augmentation import load_config, get_inference_transforms
from src.models.classifier import get_model, get_device


# ── Class names ────────────────────────────────────────────────────────────────

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


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(
    weights_path: str = "weights/best_model.pth",
    config_path:  str = "configs/train_config.yaml",
):
    config = load_config(config_path)
    device = get_device(config)
    ckpt   = torch.load(weights_path, map_location=device)
    model  = get_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    transform = get_inference_transforms(config)
    threshold = config["inference"]["confidence_threshold"]
    return model, transform, device, threshold, config


# ── Core prediction ────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict_image(
    img_rgb: np.ndarray,        # H×W×3 uint8 RGB numpy array
    model,
    transform,
    device,
    threshold: float = 0.70,
) -> dict:
    """
    Returns a dict with:
      class_id    : int   — predicted class (or -1 if uncertain)
      class_name  : str   — human-readable label
      confidence  : float — max softmax probability
      uncertain   : bool  — True if confidence < threshold
      top3        : list  — [(class_id, class_name, confidence), ...]
    """
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]

    top3_vals, top3_idx = probs.topk(3)
    top3 = [
        (int(idx), CLASS_NAMES[int(idx)], float(val))
        for idx, val in zip(top3_idx, top3_vals)
    ]

    conf      = top3[0][2]
    uncertain = conf < threshold

    return {
        "class_id":   -1 if uncertain else top3[0][0],
        "class_name": "Uncertain" if uncertain else top3[0][1],
        "confidence": conf,
        "uncertain":  uncertain,
        "top3":       top3,
    }


# ── Single image ───────────────────────────────────────────────────────────────

def predict_single(
    image_path: str,
    weights_path: str = "weights/best_model.pth",
    config_path:  str = "configs/train_config.yaml",
):
    model, transform, device, threshold, _ = load_model(weights_path, config_path)
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    result  = predict_image(img_rgb, model, transform, device, threshold)

    print(f"\n{'─'*45}")
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {result['class_name']}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    if result["uncertain"]:
        print(f"  ⚠️  Below threshold ({threshold*100:.0f}%) — marked Uncertain")
    print(f"\n  Top-3 predictions:")
    for rank, (cid, cname, conf) in enumerate(result["top3"], 1):
        print(f"    {rank}. [{cid:02d}] {cname:<35} {conf*100:.1f}%")
    print(f"{'─'*45}\n")
    return result


# ── Batch folder ───────────────────────────────────────────────────────────────

def predict_batch(
    folder_path:  str,
    output_csv:   str  = "results/predictions.csv",
    weights_path: str  = "weights/best_model.pth",
    config_path:  str  = "configs/train_config.yaml",
):
    model, transform, device, threshold, _ = load_model(weights_path, config_path)
    folder = Path(folder_path)
    exts   = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}
    images = [p for p in folder.rglob("*") if p.suffix.lower() in exts]

    print(f"\n📁 Found {len(images)} images in {folder}")
    rows = []
    for img_path in images:
        try:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            result  = predict_image(img_rgb, model, transform, device, threshold)
            rows.append({
                "file":       str(img_path),
                "class_id":   result["class_id"],
                "class_name": result["class_name"],
                "confidence": f"{result['confidence']*100:.2f}",
                "uncertain":  result["uncertain"],
            })
        except Exception as e:
            print(f"  ⚠️  Skipped {img_path.name}: {e}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "class_id", "class_name",
                                                "confidence", "uncertain"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved predictions → {output_csv}")
    return rows


# ── Overlay drawing ────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    result: dict,
    fps: float,
    threshold: float,
    crop_box: tuple,          # (x1, y1, x2, y2) of the ROI
) -> np.ndarray:
    """
    Draws prediction overlay onto a BGR OpenCV frame.
    - Green box + label if confident
    - Red box + "Uncertain" if below threshold
    - FPS counter top-right
    - Confidence bar bottom-left
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_box
    confident = not result["uncertain"]
    box_color = (0, 220, 0) if confident else (0, 0, 220)  # BGR

    # ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Label banner above the box
    label      = result["class_name"]
    conf_pct   = result["confidence"] * 100
    label_text = f"{label}  {conf_pct:.1f}%"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    banner_y1  = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 8, y1), box_color, -1)
    cv2.putText(frame, label_text, (x1 + 4, y1 - 5),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Top-3 list (bottom-left panel)
    panel_x, panel_y = 12, h - 110
    cv2.rectangle(frame, (panel_x - 4, panel_y - 20),
                  (340, h - 8), (20, 20, 20), -1)
    cv2.putText(frame, "Top-3:", (panel_x, panel_y),
                font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    for i, (cid, cname, cconf) in enumerate(result["top3"]):
        color = (0, 220, 0) if i == 0 and confident else (180, 180, 180)
        text  = f"  {i+1}. {cname[:28]:<28} {cconf*100:.1f}%"
        cv2.putText(frame, text, (panel_x, panel_y + 25 + i * 22),
                    font, 0.45, color, 1, cv2.LINE_AA)

    # Confidence bar (bottom-right)
    bar_x, bar_y, bar_w, bar_h = w - 170, h - 35, 155, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    filled = int(bar_w * result["confidence"])
    bar_color = (0, 220, 0) if confident else (0, 100, 220)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  bar_color, -1)
    cv2.putText(frame, f"Conf: {conf_pct:.1f}%", (bar_x, bar_y - 5),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Threshold line on bar
    thresh_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thresh_x, bar_y - 4), (thresh_x, bar_y + bar_h + 4),
             (255, 255, 0), 2)

    # FPS (top-right)
    fps_text = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, font, 0.6, 2)
    cv2.putText(frame, fps_text, (w - fw - 12, 28),
                font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Threshold indicator (top-left)
    cv2.putText(frame, f"Threshold: {threshold*100:.0f}%  (+/-)",
                (12, 28), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ── Webcam / video loop ────────────────────────────────────────────────────────

def run_realtime(
    source:       int | str = 0,
    weights_path: str = "weights/best_model.pth",
    config_path:  str = "configs/train_config.yaml",
):
    model, transform, device, threshold, config = load_model(weights_path, config_path)
    screenshot_dir = Path("results/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n🎥 Source opened: {cam_w}×{cam_h}")
    print("  Q = quit  |  S = screenshot  |  + = raise threshold  |  - = lower threshold\n")

    # Central ROI: a square crop (simulates placing sign in front of camera)
    roi_size = min(cam_w, cam_h) // 2
    cx, cy   = cam_w // 2, cam_h // 2
    x1, y1   = cx - roi_size // 2, cy - roi_size // 2
    x2, y2   = cx + roi_size // 2, cy + roi_size // 2
    crop_box = (x1, y1, x2, y2)

    fps_buffer = []
    result     = {"class_name": "Initialising...", "confidence": 0.0,
                  "uncertain": True, "top3": [], "class_id": -1}

    while True:
        t0  = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        # Crop ROI → predict
        roi     = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result  = predict_image(roi_rgb, model, transform, device, threshold)

        # FPS (rolling average over last 10 frames)
        fps_buffer.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        if len(fps_buffer) > 10:
            fps_buffer.pop(0)
        fps = sum(fps_buffer) / len(fps_buffer)

        # Draw and show
        display = draw_overlay(frame.copy(), result, fps, threshold, crop_box)
        cv2.imshow("Traffic Sign Classifier  (Q=quit  S=save)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = int(time.time())
            path = screenshot_dir / f"screenshot_{ts}.png"
            cv2.imwrite(str(path), display)
            print(f"  📸 Screenshot saved → {path}")
        elif key == ord("+") or key == ord("="):
            threshold = min(threshold + 0.05, 0.99)
            print(f"  Threshold → {threshold*100:.0f}%")
        elif key == ord("-"):
            threshold = max(threshold - 0.05, 0.10)
            print(f"  Threshold → {threshold*100:.0f}%")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Inference")
    parser.add_argument("--weights", default="weights/best_model.pth")
    parser.add_argument("--config",  default="configs/train_config.yaml")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",   type=str, help="Path to a single image")
    group.add_argument("--folder",  type=str, help="Folder of images → CSV")
    group.add_argument("--webcam",  action="store_true", help="Live webcam feed")
    group.add_argument("--video",   type=str, help="Path to a video file")

    parser.add_argument("--output", default="results/predictions.csv",
                        help="CSV output path for --folder mode")

    args = parser.parse_args()

    if args.image:
        predict_single(args.image, args.weights, args.config)

    elif args.folder:
        predict_batch(args.folder, args.output, args.weights, args.config)

    elif args.webcam:
        run_realtime(0, args.weights, args.config)

    elif args.video:
        run_realtime(args.video, args.weights, args.config)