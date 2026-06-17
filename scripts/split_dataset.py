"""
split_dataset.py — One-time setup script.

Copies the F25 training dataset into your Sum26 folder and creates
a proper 80/20 train/val split. Run this ONCE before training.

After running, your Sum26 datasets/ folder will look like:
    datasets/
        images/
            train/   (~80% of F25 images)
            val/     (~20% of F25 images)
        labels/
            train/   (matching labels)
            val/     (matching labels)

UPDATE THE PATHS below before running.
"""

import os
import shutil
import random

# ── UPDATE THESE PATHS ────────────────────────────────────────────────────────

# Source: F25 dataset (read-only — do not modify)
F25_IMAGES_TRAIN = "/content/drive/MyDrive/F25_VisualDist_Extern/ALLCODE-DOWNLOADTHISDIRECTLY/Project Materials/datasets/images/train"
F25_LABELS_TRAIN = "/content/drive/MyDrive/F25_VisualDist_Extern/ALLCODE-DOWNLOADTHISDIRECTLY/Project Materials/datasets/labels/train"

# Destination: your Sum26 working folder (you own this)
SUM26_DATASETS = "/content/drive/MyDrive/Sum26_VisualDist_Extern/Visual_Distractions_YOLOv11/datasets"

# Split ratio
VAL_RATIO = 0.2
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────

IMG_TRAIN = os.path.join(SUM26_DATASETS, "images", "train")
IMG_VAL   = os.path.join(SUM26_DATASETS, "images", "val")
LBL_TRAIN = os.path.join(SUM26_DATASETS, "labels", "train")
LBL_VAL   = os.path.join(SUM26_DATASETS, "labels", "val")

def main():
    # Safety check — don't overwrite existing data
    if os.path.exists(IMG_TRAIN) and len(os.listdir(IMG_TRAIN)) > 0:
        print("ERROR: datasets/ already exists and has files. Aborting to avoid overwrite.")
        print("Delete the existing datasets/ folder first if you want to re-run this script.")
        return

    for folder in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        os.makedirs(folder, exist_ok=True)

    # Copy all images from F25
    print("Copying images from F25...")
    all_imgs = [f for f in os.listdir(F25_IMAGES_TRAIN) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for f in all_imgs:
        shutil.copy(os.path.join(F25_IMAGES_TRAIN, f), os.path.join(IMG_TRAIN, f))
    print(f"  Copied {len(all_imgs)} images.")

    # Copy all labels from F25
    print("Copying labels from F25...")
    all_labels = [f for f in os.listdir(F25_LABELS_TRAIN) if f.endswith(".txt")]
    for f in all_labels:
        shutil.copy(os.path.join(F25_LABELS_TRAIN, f), os.path.join(LBL_TRAIN, f))
    print(f"  Copied {len(all_labels)} labels.")

    # Create val split
    random.seed(RANDOM_SEED)
    random.shuffle(all_imgs)
    val_count = int(len(all_imgs) * VAL_RATIO)
    val_imgs  = all_imgs[:val_count]

    print(f"\nCreating val split ({int(VAL_RATIO*100)}%)...")
    missing_labels = []
    for img_file in val_imgs:
        # Move image to val
        shutil.move(os.path.join(IMG_TRAIN, img_file), os.path.join(IMG_VAL, img_file))

        # Move matching label to val
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_src  = os.path.join(LBL_TRAIN, lbl_file)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(LBL_VAL, lbl_file))
        else:
            missing_labels.append(img_file)

    print(f"\nDone!")
    print(f"  Train: {len(all_imgs) - val_count} images")
    print(f"  Val:   {val_count} images")
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} val images had no matching label file:")
        for f in missing_labels:
            print(f"    - {f}")

if __name__ == "__main__":
    main()
