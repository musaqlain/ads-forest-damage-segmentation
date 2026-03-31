# ADS Polygon Alignment Pipeline

Aligns U.S. Forest Service Aerial Detection Survey (ADS) polygons with NAIP imagery using a hybrid energy optimization adapted from ProximityAlign (Cherif et al., ISPRS 2024).

## Files

| File | Role | Description |
|------|------|-------------|
| `build_seed_data.py` | Data preparation | Downloads year-matched NAIP tiles, generates DeepForest masks, outputs `paired_samples_2024.pkl` |
| `coarse_align.py` | Core algorithm | ProximityAlign hybrid energy alignment (contrast + contour distance + OOB penalty) |
| `evaluate_coarse.py` | Evaluation | Runs alignment on seed data, computes IoU/centroid error, generates visualization grids |
| `augmentation.py` | Training data | `SyntheticDisplacer` generates unlimited displaced polygon pairs from seed samples |
| `demo_weak_augmentation.py` | Visualization | Demonstrates curriculum augmentation (coarse, medium, fine displacements) |
| `transforms.py` | Utilities | Polygon rasterization, NAIP normalization, coordinate transforms |
| `requirements.txt` | Dependencies | PyTorch, Rasterio, GeoPandas, DeepForest, etc. |

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build seed dataset (requires .gdb and .gpkg in data/)
python build_seed_data.py

# 3. Evaluate Stage 1 alignment
python evaluate_coarse.py
# -> outputs in outputs/coarse_eval/

# 4. Demo augmentation engine
python demo_weak_augmentation.py
# -> outputs in data/weak_aug_demo/
```

## Pipeline Flow

```
build_seed_data.py -> data/paired_samples_2024.pkl
                          |                |
               evaluate_coarse.py    demo_weak_augmentation.py
               (alignment eval)      (augmentation demo)
                     |
         outputs/coarse_eval/
         +-- coarse_eval_grid.png
         +-- coarse_eval_summary.png
         +-- coarse_eval_results.csv
```

Both `evaluate_coarse.py` and `demo_weak_augmentation.py` depend on `coarse_align.py`, `augmentation.py`, and `transforms.py` as shared modules.
