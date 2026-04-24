# 🛰️ Satellite Image Segmentation & Change Detection

A multi-approach research project for detecting and segmenting land-cover changes in multitemporal satellite imagery. Three distinct methods are implemented and compared — from a classical pixel-differencing baseline to an unsupervised spatial-context model and a transformer-based deep learning approach.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Approaches](#approaches)
  - [1. Pixel-wise Differencing (Baseline)](#1-pixel-wise-differencing-baseline)
  - [2. SiROC — Unsupervised Spatial Context Change Detection](#2-siroc--unsupervised-spatial-context-change-detection)
  - [3. SegFormer — Transformer-based Semantic Segmentation](#3-segformer--transformer-based-semantic-segmentation)
  - [4. SAMGeo — Segment Anything for Geospatial Imagery](#4-samgeo--segment-anything-for-geospatial-imagery)
- [Datasets](#datasets)
- [Results Summary](#results-summary)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [References](#references)

---

## Overview

This project investigates satellite image change detection through a comparison of classical and modern deep-learning techniques. The goal is to identify **what changed**, **where it changed**, and **what land-cover class** the region belongs to, using bi-temporal satellite image pairs.

| Method                  | Type             | Supervision  | Dataset             |
| ----------------------- | ---------------- | ------------ | ------------------- |
| Pixel-wise Differencing | Classical        | Unsupervised | OSCD (Onera)        |
| SiROC                   | Statistical / ML | Unsupervised | OSCD (Onera)        |
| SegFormer               | Deep Learning    | Supervised   | SECOND              |
| SAMGeo                  | Foundation Model | Zero-shot    | Custom / Geospatial |

---

## Repository Structure

```
Satellite-Image-Segmentation-and-change-detection/
│
├── Onera Dataset/              # OSCD dataset (Onera Satellite Change Detection)
│   ├── Images/
│   ├── splits/
│   ├── TestLabels/
│   └── TrainLabels/
│
├── SECOND_train_set/           # SECOND dataset for semantic change detection
│   ├── im1/                    # Pre-event images
│   ├── im2/                    # Post-event images
│   ├── label1/                 # Land-cover labels for im1
│   ├── label2/                 # Land-cover labels for im2
│   └── pair/                   # Paired image metadata
│
├── Pixel wise diff/            # Approach 1: Classical baseline
│   ├── pixel_diff.py           # Main pipeline script
│   ├── pixel_diff.ipynb        # Exploratory notebook
│   ├── requirements.txt
│   └── results/                # Output images
│
├── SiROC_change/               # Approach 2: Unsupervised SiROC method
│   ├── method.py               # Core SiROC algorithm
│   ├── metrics.py              # Evaluation metrics
│   ├── predict_new.py          # Inference script
│   ├── requirements.txt
│   └── data/
│
├── SegFormer/                  # Approach 3: Transformer-based segmentation
│   ├── config.py               # Hyperparameters and paths
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation on any split
│   ├── test.py                 # Visual prediction & export
│   ├── models/
│   │   └── segformer_model.py  # SegFormerChange model wrapper
│   ├── utils/
│   │   ├── dataset.py          # ChangeDataset loader
│   │   ├── metrics.py          # mIoU, pixel accuracy
│   │   └── visualtization.py   # Prediction visualisation helpers
│   ├── runs/                   # Saved checkpoints
│   └── outputs/                # Prediction output images
│
└── samgeo/                     # Approach 4: SAMGeo notebook
    └── Samgeo_change_detection.ipynb
```

---

## Approaches

### 1. Pixel-wise Differencing (Baseline)

**Location:** `Pixel wise diff/`

A classical, training-free baseline that computes binary change masks using per-pixel intensity differences between image pairs.

**Pipeline:**

```
Pre-event image ──┐
                  ├──► Histogram Matching ──► |post_norm − pre| ──► Grayscale ──► Normalize ──► Threshold (τ=0.15) ──► Morphological Opening ──► Binary Mask
Post-event image ─┘
```

1. **Histogram matching** — normalises the post-event image's colour distribution to match the pre-event image to reduce radiometric differences.
2. **Pixel-wise absolute difference** — computes `|post_norm − pre|` across all three RGB channels.
3. **Grayscale conversion** — averages channel residuals into a single difference map.
4. **Thresholding** — applies a fixed threshold (`τ = 0.15`) to produce a binary change mask.
5. **Morphological opening** — removes noise and small spurious regions with a 5×5 kernel.
6. **Evaluation** — computes Precision, Recall, F1, IoU vs. ground-truth labels.

**Run:**

```bash
cd "Pixel wise diff"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pixel_diff.py
```

**Outputs in `results/`:**

| File              | Description                                        |
| ----------------- | -------------------------------------------------- |
| `before.png`      | Pre-event image                                    |
| `after.png`       | Post-event image                                   |
| `overlay.png`     | Post-event image with detected changes highlighted |
| `pred_change.png` | Binary predicted change mask                       |

---

### 2. SiROC — Unsupervised Spatial Context Change Detection

**Location:** `SiROC_change/`

Implementation of the IEEE TGRS 2021 paper **"Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images"** by Kondmann et al.

> **Key Idea:** SiROC (Sibling Regression for Optical Change detection) models each pixel as a linear combination of its distant spatial neighbours. If the relationship between a pixel and its neighbours changes between two time periods, the pixel is flagged as changed. No labelled training data is required.

**How it works:**

1. For each pixel in the pre-event image, a linear regression model is fit using its spatially distant neighbours (excluding a close inner ring).
2. The same model is applied to the post-event image. Pixels whose predictions diverge significantly are marked as changed.
3. Predictions are ensembled over mutually exclusive neighbourhoods.
4. Otsu or Triangle thresholding converts the change map to a binary mask.
5. Morphological operations transition from pixel-level to object-level changes.

**Run:**

```bash
cd SiROC_change
pip install -r requirements.txt
python predict_new.py
```

**Dataset:** [OSCD — Onera Satellite Change Detection](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)

> Move `train.txt` and `test.txt` from the image directory into a `splits/` subfolder at the root of the Onera Dataset.

**Reference:**

```bibtex
@article{kondmann2021spatial,
  title={Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images},
  author={Kondmann, Lukas and Toker, Aysim and Saha, Sudipan and Sch{\"o}lkopf, Bernhard and Leal-Taix{\'e}, Laura and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
```

---

### 3. SegFormer — Transformer-based Semantic Segmentation

**Location:** `SegFormer/`

A supervised deep learning approach that fine-tunes a **SegFormer-B0** backbone (pre-trained on ADE20K) for **multi-class semantic change detection** on the SECOND dataset.

**Model Architecture:**

- **Backbone:** `nvidia/segformer-b0-finetuned-ade-512-512` (Mix Transformer encoder)
- **Input:** Concatenated 6-channel image `[im1 | im2]` (RGB pre + RGB post)
- **Output:** Per-pixel semantic class label (7 classes)
- **Loss:** Weighted cross-entropy (class frequency-balanced)
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-4`)

**Classes:**

| ID  | Class          | Colour     |
| --- | -------------- | ---------- |
| 0   | non_change     | Black      |
| 1   | low_vegetation | Green      |
| 2   | nvg_surface    | Coral      |
| 3   | tree           | Dark Green |
| 4   | water          | Blue       |
| 5   | building       | Red        |
| 6   | playground     | Cyan       |

**Setup:**

```bash
cd SegFormer
pip install torch torchvision transformers opencv-python numpy
```

**Training:**

```bash
python train.py \
  --epochs 30 \
  --batch_size 4 \
  --img_size 256 \
  --lr 1e-4
```

**Evaluation:**

```bash
python evaluate.py --split val --ckpt runs/segformer_change_best.pt
```

**Visual Prediction:**

```bash
# On a dataset split
python test.py --split val --num_images 10 --with_legend --out outputs/pred.png

# On a custom image pair
python test.py --im1 path/to/before.png --im2 path/to/after.png --out outputs/result.png
```

**Training Outputs (in `runs/`):**

| File                       | Description                          |
| -------------------------- | ------------------------------------ |
| `segformer_change_best.pt` | Best checkpoint (highest val mIoU)   |
| `segformer_change_last.pt` | Latest epoch checkpoint              |
| `train_history.json`       | Per-epoch loss, pixel accuracy, mIoU |

---

### 4. SAMGeo — Segment Anything for Geospatial Imagery

**Location:** `samgeo/`

An exploratory notebook demonstrating zero-shot geospatial change detection using **SAMGeo**, a geospatial adaptation of Meta's Segment Anything Model (SAM).

**Notebook:** `Samgeo_change_detection.ipynb`

SAMGeo enables prompt-based segmentation of satellite tiles without any task-specific training. The notebook explores:

- Automatic mask generation on satellite tiles
- Change detection via segment-level comparison
- Integration with Leafmap for interactive visualisation

---

## Datasets

### OSCD — Onera Satellite Change Detection

- **Source:** [IEEE Dataport](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)
- **Used by:** Pixel-wise Differencing, SiROC
- **Content:** 24 pairs of multispectral Sentinel-2 images; binary change labels
- **Resolution:** ~10m/pixel

**Expected layout:**

```
Onera Dataset/
├── Images/
│   └── Train/
│       └── <city>/pair/img1.png, img2.png
├── splits/
│   ├── train.txt
│   └── test.txt
├── TestLabels/
│   └── <city>/cm/cm.png
└── TrainLabels/
    └── <city>/cm/cm.png
```

### SECOND — Semantic Change Detection Dataset

- **Used by:** SegFormer
- **Content:** Aerial image pairs with per-pixel semantic land-cover labels for both time periods
- **Classes:** 6 land-cover types + no-change

**Expected layout:**

```
SECOND_train_set/
├── im1/        # Pre-event RGB images
├── im2/        # Post-event RGB images
├── label1/     # Semantic labels for im1
├── label2/     # Semantic labels for im2
└── pair/
```

---

## Results Summary

| Method          | Dataset          | Precision | Recall | F1-Score      | Notes                                   |
| --------------- | ---------------- | --------- | ------ | ------------- | --------------------------------------- |
| Pixel-wise Diff | OSCD (Abu Dhabi) | ~0.02     | ~0.08  | ~0.034        | High FP rate; sensitive to illumination |
| SiROC           | OSCD             | —         | —      | Competitive\* | Unsupervised; no labelled data needed   |
| SegFormer       | SECOND           | —         | —      | —             | Tracks mIoU & pixel acc per epoch       |

> \*Refer to the [SiROC paper](https://ieeexplore.ieee.org/document/9627707) for benchmark numbers across four datasets.

---

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended for SegFormer training (CUDA or MPS)
- `git clone` the repository

### Quick Start per Approach

```bash
# Clone
git clone https://github.com/<your-username>/Satellite-Image-Segmentation-and-change-detection.git
cd Satellite-Image-Segmentation-and-change-detection

# ── Approach 1: Pixel Differencing ─────────────────────────────────────
cd "Pixel wise diff"
pip install -r requirements.txt
python pixel_diff.py

# ── Approach 2: SiROC ──────────────────────────────────────────────────
cd ../SiROC_change
pip install -r requirements.txt
python predict_new.py

# ── Approach 3: SegFormer ──────────────────────────────────────────────
cd ../SegFormer
pip install torch torchvision transformers opencv-python numpy
python train.py
python evaluate.py --split val
python test.py --split val --with_legend --out outputs/pred.png
```

---

## Requirements

### Pixel-wise Differencing

| Package         | Purpose                      |
| --------------- | ---------------------------- |
| `opencv-python` | Image I/O, morphological ops |
| `scikit-image`  | Histogram matching           |
| `matplotlib`    | Visualisation                |
| `numpy`         | Array operations             |
| `scikit-learn`  | Confusion matrix             |
| `seaborn`       | Heatmap plots                |

### SiROC

| Package                | Purpose                  |
| ---------------------- | ------------------------ |
| `torch`, `torchvision` | Tensor operations        |
| `opencv-python`        | Thresholding, morphology |
| `numpy`                | Array operations         |
| `matplotlib`           | Plotting                 |
| `scikit-learn`         | Metrics                  |

### SegFormer

| Package                | Purpose                        |
| ---------------------- | ------------------------------ |
| `torch`, `torchvision` | Deep learning framework        |
| `transformers`         | HuggingFace SegFormer backbone |
| `opencv-python`        | Image I/O                      |
| `numpy`                | Array operations               |

---

## References

1. **SiROC:** Kondmann, L., Toker, A., Saha, S., Schölkopf, B., Leal-Taixé, L., & Zhu, X. X. (2021). _Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images._ IEEE Transactions on Geoscience and Remote Sensing. [DOI](https://ieeexplore.ieee.org/document/9627707)

2. **SegFormer:** Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). _SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers._ NeurIPS 2021.

3. **SAMGeo:** Wu, Q., & Osco, L. (2023). _samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM)._ Journal of Open Source Software.

4. **OSCD Dataset:** Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018). _Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks._ IGARSS 2018.

5. **SECOND Dataset:** Yang, K., Xia, G. S., Liu, Z., Du, B., Yang, W., Pelillo, M., & Zhang, L. (2021). _Asymmetric Siamese Networks for Semantic Change Detection in Aerial Images._ IEEE TGRS.

---
