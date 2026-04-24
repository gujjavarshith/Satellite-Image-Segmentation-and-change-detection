# Pixel-Based Change Detection

Classical baseline for binary change detection on the ONERA Satellite Change Detection dataset. Computes pixel-wise absolute differences between histogram-matched image pairs and applies thresholding with morphological post-processing to produce a binary change mask.

## Pipeline

1. **Load** pre-event and post-event RGB images
2. **Histogram matching** — normalise the post-event image's colour distribution to match the pre-event image
3. **Pixel-wise differencing** — compute `|post_norm − pre|` across all three channels
4. **Grayscale conversion** — average the RGB channel residuals into a single difference map
5. **Normalise** the difference map to `[0, 1]`
6. **Threshold** at `τ = 0.15` to produce a binary change mask
7. **Morphological opening** (5×5 kernel) to remove noise and small spurious regions
8. **Evaluate** against ground-truth binary labels — computes TP, FP, FN, TN, Precision, Recall, F1, IoU
9. **Visualise** confusion matrix and error map (TP=green, FP=red, FN=blue)

## Dataset

ONERA Satellite Change Detection (OSCD) dataset. Place it at the following relative path from this folder:

```
../Onera Dataset/
    Images/Train/abudhabi/pair/img1.png
    Images/Train/abudhabi/pair/img2.png
    TrainLabels/abudhabi/cm/cm.png
```

## Setup

```bash
python3 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
venv\Scripts\Activate.ps1
python pixel_diff.py
```

## Outputs

| File                      | Description                                        |
| ------------------------- | -------------------------------------------------- |
| `results/before.png`      | Pre-event image                                    |
| `results/after.png`       | Post-event image                                   |
| `results/overlay.png`     | Post-event image with detected changes highlighted |
| `results/pred_change.png` | Binary predicted change mask                       |

## Results (ONERA Abu Dhabi)

| Metric    | Value  |
| --------- | ------ |
| Precision | ~0.02  |
| Recall    | ~0.08  |
| F1-Score  | ~0.034 |

The low precision reflects high false-positive contamination — pixel differencing is sensitive to any illumination or sensor difference not captured by global histogram alignment.

## Dependencies

| Package         | Purpose                      |
| --------------- | ---------------------------- |
| `opencv-python` | Image I/O, morphological ops |
| `scikit-image`  | Histogram matching           |
| `matplotlib`    | Visualisation                |
| `numpy`         | Array operations             |
| `scikit-learn`  | Confusion matrix             |
| `seaborn`       | Heatmap plots                |
