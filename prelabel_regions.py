"""
prelabel_regions.py  — (optional) auto-DRAFT damage polygons for Labelme
========================================================================
Generates ROUGH candidate "damage" polygons from image signals so you don't start
from a blank tile in Labelme — you just FIX the drafts (delete wrong ones, drag
vertices). Writes one Labelme JSON per tile into data/seed30cm/labelme/.

Signals used (see PROJECT_MAP.md / the measured AUC table):
  * redness  = (R - G)/(R + G)   — dead conifers turn red/brown (R > G)
  * NDVI     = (NIR - R)/(NIR + R) if a NIR band is present (nir/<id>.png) — dead = low
  * brightness anomaly           — bare/gray patches (clearcut, bleached snags)
The draft is constrained to NEAR the rough ADS polygon (priors/<id>.png), since that
is where the survey says damage is.

HONEST EXPECTATIONS (from validation): drafts are GOOD where damage is a distinct
red/brown patch (you'll just tweak edges), and POOR where the "damage" is diffuse
thinning of still-green canopy (you'll redraw those). It saves the finding + first
draft, not the judgement. YOU still review every tile.

Run (in ads_env), then annotate:
  python prelabel_regions.py
  labelme data/seed30cm/images --output data/seed30cm/labelme --labels damage,prior
Then convert as usual:  python labelme_to_masks.py

Config knobs below. Set SEED30CM_DIR env var to point elsewhere.
"""
import os
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import uniform_filter, binary_opening, binary_closing

DATA = Path(os.environ.get("SEED30CM_DIR", "data/seed30cm"))
IMG_DIR = DATA / "images"
PRIOR_DIR = DATA / "priors"
NIR_DIR = DATA / "nir"                 # optional single-band NIR PNGs (if you fetched them)
LABEL_DIR = DATA / "labelme"

# --- tuning ---
PCT_THR = 65.0            # draft the reddest/most-stressed (100-PCT)% of pixels within the ROI
REDNESS_FLOOR = 0.005     # ...but only where R actually exceeds G (avoids flagging healthy tiles)
NEAR_PRIOR_FRAC = 0.15    # dilate the ADS prior by this fraction of the tile before masking
MIN_BLOB_FRAC = 0.004     # drop candidate blobs smaller than this fraction of the tile
SIMPLIFY_FRAC = 0.006     # polygon simplification (bigger = fewer vertices)
OVERWRITE = False         # True = replace existing JSONs (WARNING: discards your annotations)


def damage_score(rgb, nir=None):
    """Per-pixel 'looks like damage' score (higher = more likely dead/brown)."""
    R, G, B = [rgb[..., i].astype(np.float32) for i in range(3)]
    score = (R - G) / (R + G + 1.0)                   # redness: dead conifers => positive
    if nir is not None:
        N = nir.astype(np.float32)
        ndvi = (N - R) / (N + R + 1.0)
        score = score + 0.5 * np.clip(0.35 - ndvi, 0.0, None)   # add NDVI stress (low ndvi=dead)
    return score


def candidate_polygons(rgb, prior_bool, nir=None):
    """Return list of polygons (each list of (x,y)) drafting the damage regions."""
    H, W = rgb.shape[:2]
    score = damage_score(rgb, nir)

    # region of interest = ADS prior dilated by a margin (damage can spill just outside)
    if prior_bool is not None and prior_bool.any():
        k = max(3, int(NEAR_PRIOR_FRAC * max(H, W)))
        roi = cv2.dilate(prior_bool.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    else:
        roi = np.ones((H, W), bool)

    # adaptive threshold: the reddest pixels *within* the ROI (but must be genuinely browning)
    thr = max(REDNESS_FLOOR, float(np.percentile(score[roi], PCT_THR)))
    hot = (score > thr) & roi
    hot = binary_opening(hot, iterations=2)           # kill specks
    hot = binary_closing(hot, iterations=3)           # fill holes / merge nearby

    cnts, _ = cv2.findContours(hot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_BLOB_FRAC * H * W
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = SIMPLIFY_FRAC * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) >= 3:
            polys.append(ap.astype(float).tolist())
    return polys


def _load(stem):
    rgb = np.array(Image.open(IMG_DIR / f"{stem}.png").convert("RGB"))
    pp = PRIOR_DIR / f"{stem}.png"
    prior = (np.array(Image.open(pp).convert("L")) > 128) if pp.exists() else None
    npp = NIR_DIR / f"{stem}.png"
    nir = np.array(Image.open(npp).convert("L")) if npp.exists() else None
    return rgb, prior, nir


def main():
    if not IMG_DIR.exists():
        print(f"No images at {IMG_DIR}. Run build_30cm_seed_tiles.py first.")
        return
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(IMG_DIR.glob("*.png")):
        out = LABEL_DIR / f"{img.stem}.json"
        if out.exists() and not OVERWRITE:
            print(f"  {img.stem}: JSON exists, skipping (OVERWRITE=True to replace)")
            continue
        rgb, prior, nir = _load(img.stem)
        polys = candidate_polygons(rgb, prior, nir)
        H, W = rgb.shape[:2]
        shapes = [{"label": "damage", "points": p, "group_id": None,
                   "shape_type": "polygon", "flags": {}} for p in polys]
        rel = os.path.relpath(img, LABEL_DIR).replace(os.sep, "/")
        out.write_text(json.dumps({
            "version": "5.5.0", "_autodraft": True, "flags": {}, "shapes": shapes,
            "imagePath": rel, "imageData": None, "imageHeight": H, "imageWidth": W,
        }, indent=2), encoding="utf-8")   # _autodraft => labelme_to_masks skips until you review+save
        n += 1
        print(f"  {img.stem}: drafted {len(polys)} candidate region(s)  "
              f"(nir={'yes' if nir is not None else 'no'})")
    print(f"\nDrafted {n} tile(s) into {LABEL_DIR}. Open in Labelme and FIX the drafts:")
    print("  labelme data/seed30cm/images --output data/seed30cm/labelme --labels damage,prior")


if __name__ == "__main__":
    main()
