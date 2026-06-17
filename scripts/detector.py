"""
detector.py — Run YOLOv11 inference on Street View images.

Updated from YOLOv5 (S26) to YOLOv11 (Sum26) using the Ultralytics Python API.
Usage:
    python detector.py --model path/to/best.pt --data path/to/images/
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# === CONFIGURATION ===
DEFAULT_MODEL_PATH = "best.pt"
DEFAULT_DATA_DIR   = "data"
DEFAULT_OUTPUT_CSV = "detections.csv"
CONF_THRESH        = 0.25
IMG_SIZE           = 640


def run_detection(model_path: str, data_dir: str, output_csv: str):
    # 1. Load the YOLOv11 model
    print(f"Loading YOLOv11 model from {model_path}...")
    model = YOLO(model_path)

    # 2. Collect images
    data_path = Path(data_dir)
    image_files = (
        list(data_path.glob("*.jpg"))
        + list(data_path.glob("*.jpeg"))
        + list(data_path.glob("*.png"))
    )

    if not image_files:
        print(f"No images found in {data_dir}.")
        return

    print(f"Processing {len(image_files)} images...")
    all_detections = []

    for img_path in image_files:
        print(f"  Scanning {img_path.name}...")

        # Parse lat/lng from filename: sv_lat_lng_degHEADING.jpg
        try:
            parts = img_path.stem.split("_")
            lat = float(parts[1])
            lng = float(parts[2])
        except (IndexError, ValueError):
            print(f"    Warning: Could not parse coordinates from {img_path.name}. Using 0.0, 0.0")
            lat, lng = 0.0, 0.0

        # Run inference
        results = model.predict(source=str(img_path), conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)

        # Parse detections
        class_names = model.names  # {0: 'tree', 1: 'pole', ...}
        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls)
                conf    = round(float(box.conf), 4)
                xmin, ymin, xmax, ymax = [round(float(v), 2) for v in box.xyxy[0].tolist()]

                all_detections.append({
                    "image_name": img_path.name,
                    "lat":        lat,
                    "lng":        lng,
                    "class":      class_names[cls_id],
                    "confidence": conf,
                    "xmin":       xmin,
                    "ymin":       ymin,
                    "xmax":       xmax,
                    "ymax":       ymax,
                })

        print(f"    Found {len(results[0].boxes)} objects at ({lat}, {lng}).")

    # 3. Save to CSV
    if all_detections:
        df = pd.DataFrame(all_detections)
        df.to_csv(output_csv, index=False)
        print(f"\nDetection complete! {len(all_detections)} total detections saved to {output_csv}")
    else:
        print("\nNo objects detected in any images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv11 inference on Street View images.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to best.pt")
    parser.add_argument("--data",  default=DEFAULT_DATA_DIR,   help="Directory of input images")
    parser.add_argument("--out",   default=DEFAULT_OUTPUT_CSV, help="Output CSV filename")
    args = parser.parse_args()

    run_detection(args.model, args.data, args.out)
