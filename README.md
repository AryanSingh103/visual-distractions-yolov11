# Visual Distractions at Railroad Grade Crossings — YOLOv11

**Rutgers MBS Externship | Collaborative Solutions | Summer 2026**

Detects visual distractions at railroad grade crossings using Google Street View imagery and a custom-trained YOLOv11 model. Produces a risk score per crossing based on the density and type of detected objects.

---

## Project Structure

```
visual-distractions-yolov11/
├── notebooks/
│   └── train_yolov11.ipynb       # Main Colab training notebook
├── scripts/
│   ├── dataset_generation.py     # Fetches Street View images via Google Maps API
│   ├── detector.py               # Runs YOLOv11 inference on images
│   └── split_dataset.py          # Copies F25 data to Sum26 and creates train/val split
├── configs/
│   └── data.yaml                 # YOLOv11 dataset config (update path before training)
├── .gitignore
└── README.md
```

> **Note:** `datasets/` and `runs/` live in Google Drive (Sum26), not this repo.

---

## Detection Classes

| ID | Class     | Risk Weight |
|----|-----------|-------------|
| 0  | tree      | 0.35        |
| 1  | pole      | 0.15        |
| 2  | building  | 0.30        |
| 3  | billboard | 0.10        |
| 4  | sign      | 0.10        |

---

## Setup (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-username>/visual-distractions-yolov11.git
%cd visual-distractions-yolov11

!pip install ultralytics -q
```

Then open `notebooks/train_yolov11.ipynb` and follow the cells in order.

---

## First-Time Data Setup

Before training, run `scripts/split_dataset.py` once to copy the F25 dataset
into your Sum26 Drive folder and create a proper 80/20 train/val split.

Update the paths at the top of the script to match your Drive structure.

---

## Team

Aryan Singh, Shreyans Bhuyan, Asmi Kaushal, Jonathan Roman, Kokulnath Ramasamy

MBS Advisors: Brian Petrus, Ryan Maiorano
Program Mentors: Dr. John Betak, Dr. Stephen, Michael White MS
