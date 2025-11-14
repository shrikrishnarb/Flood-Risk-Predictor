# FloodRisk AI — Satellite Flood Segmentation + Risk Estimation (Minimal End‑to‑End)

A compact, end‑to‑end project that **segments flooded areas from satellite imagery** and **estimates population exposure (risk)**. It’s intentionally simple, with **few files** but complete coverage: training, evaluation, inference, a **FastAPI** service, **graphs**, and **Docker**.

---

## ✨ Highlights

- **Single-file core** (`floodrisk.py`): dataset, U‑Net, train/eval/infer, risk estimation
- **API** (`app.py`): `/segment` and `/risk` endpoints with PNG outputs (base64)
- **Synthetic demo** (`test_graphs.py`): no external data required; generates example plots
- **Minimal setup**: `requirements.txt`, `Dockerfile`, `.gitignore`
- **Pretty README graphs**: PR curve, IoU histogram, overlays, risk heatmaps

---

## 📁 Repository Structure
FloodRiskPredictor/
    ├─ floodrisk.py         # all ML + risk logic in one place (U-Net, train/eval/infer/risk)
    ├─ app.py               # FastAPI app exposing /segment and /risk
    ├─ test_graphs.py       # synthetic demo that generates graphs/overlays
    ├─ requirements.txt
    ├─ Dockerfile
    ├─ .gitignore
    └─ README.md

---

## 🔧 Environment Setup

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt  # If you don’t need geospatial features yet, you may comment out rasterio, geopandas, shapely in requirements.txt.
```

## Quickstart (Synthetic Data)
```bash
python test_graphs.py
```
This creates:
- outputs/synth_image.png — synthetic satellite-like image
- outputs/synth_mask.png — synthetic ground-truth mask
- outputs/synth_overlay.png — mask overlay visualization
- outputs/synth_pr.png — precision–recall curve
- outputs/synth_risk.png — risk heatmap with synthetic population

## Training, Evaluation, Inference, Risk
1. Prepare your data
Assume aligned image/mask pairs:
data/train/images/*.png  # or .jpg (RGB or single-channel converted to RGB)
data/train/masks/*.png   # binary masks (0/255), same name ordering as images

Keep images small (e.g., 256×256) for quick experiments. For SAR, normalize appropriately before saving as PNG.

2. Train
```bash
python floodrisk.py train \
  --images data/train/images \
  --masks  data/train/masks \
  --epochs 5 --batch 4 --lr 1e-3 --size 256 --base 32
```
- Saves model weights to models/unet.pt
- Uses BCE + Dice hybrid loss by default

3. Evaluate
```bash
python floodrisk.py eval \
  --images data/val/images \
  --masks  data/val/masks \
  --weights models/unet.pt \
  --size 256 --base 32
  ```
Outputs to outputs/:
- pr_curve.png — mean PR curve across thresholds
- iou_hist.png — IoU distribution across validation samples

4. Inference (single image)
```bash
python floodrisk.py infer \
  --image path/to/image.png \
  --weights models/unet.pt \
```
Outputs:
- outputs/mask.png — predicted binary mask
- outputs/overlay.png — mask overlay on the input image

5. Risk Estimation (flood × population)
You can provide a population raster (GeoTIFF) or a CSV grid (rows × cols). If not provided, a synthetic gradient is used.
```bash
python floodrisk.py risk \
  --image path/to/image.png \
  --weights models/unet.pt \
  --population path/to/population.tif \
  --size 256 --base 32 --threshold 0.5
```
Outputs:
- outputs/risk_heatmap.png — overlay of population density and flood mask
- outputs/risk_report.json — { "risk_score": <float> }

Make sure the population layer aligns with the image grid. The script will resample if shapes differ.

## FastAPI Service
