"""
prelabel_learned.py  — auto-DRAFT damage polygons with a classifier LEARNED from your labels
============================================================================================
This is the smarter cousin of prelabel_regions.py. Instead of a hand-tuned redness threshold,
it trains a small RandomForest on the tiles you HAVE labeled (per-pixel features: RGB, NDVI,
redness, brightness, texture), then predicts damage on the UNLABELED tiles and writes rough
Labelme "damage" polygons for you to fix. It improves automatically as you label more tiles.

It also runs a leave-one-out check on your labeled tiles and prints the mean IoU, so you can
watch the drafts get better as your label count grows (rough guide: usable > ~0.4).

Run (in ads_env), AFTER you've labeled a handful of tiles:
  python prelabel_learned.py
  labelme data/seed30cm/images --output data/seed30cm/labelme --labels damage,prior
  python labelme_to_masks.py

Needs: scikit-learn (already in ads_env). Set SEED30CM_DIR to point elsewhere.
"""
import os
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import uniform_filter, binary_opening, binary_closing
from sklearn.ensemble import RandomForestClassifier

DATA = Path(os.environ.get("SEED30CM_DIR", "data/seed30cm"))
IMG_DIR, PRIOR_DIR, NIR_DIR = DATA / "images", DATA / "priors", DATA / "nir"
MASK_DIR, LABEL_DIR = DATA / "masks", DATA / "labelme"

WORK = 512               # compute features at this resolution (speed); polygons scaled back up
N_PER_TILE = 6000        # pixels sampled per class per labeled tile
PROB_THR = 0.5           # damage probability threshold
NEAR_PRIOR_FRAC = 0.15   # restrict predictions to the ADS prior dilated by this (cuts false positives)
MIN_BLOB_FRAC = 0.004
SIMPLIFY_FRAC = 0.006
OVERWRITE = False        # True = also re-draft REVIEWED tiles (normally leave False)
DRAFT_VERSION = "5.5.0"  # prelabel stamp; a JSON still at this version / with _autodraft = NOT yet reviewed
LOO_MAX_TILES = 20       # leave-one-out quality gauge trains ONE RandomForest PER held-out tile, so full
                         # LOO is O(#labels) forests -> at 142 labels it ran ~9h and never finished. Cap
                         # the gauge to this many random tiles (fast, still representative). 0 = skip it.


def _load(stem, size=WORK):
    rgb = np.array(Image.open(IMG_DIR / f"{stem}.png").convert("RGB").resize((size, size))) / 255.0
    npp = NIR_DIR / f"{stem}.png"
    nir = (np.array(Image.open(npp).convert("L").resize((size, size))) / 255.0
           if npp.exists() else None)
    return rgb.astype(np.float32), (nir.astype(np.float32) if nir is not None else None)


def features(rgb, nir):
    """(H,W,F) per-pixel feature stack: RGB, NDVI, redness, brightness, texture."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    redness = (R - G) / (R + G + 1e-3)
    bright = (R + G + B) / 3.0
    tex = np.sqrt(np.maximum(uniform_filter(bright ** 2, 9) - uniform_filter(bright, 9) ** 2, 0))
    ndvi = ((nir - R) / (nir + R + 1e-3)) if nir is not None else np.zeros_like(R)
    return np.stack([R, G, B, ndvi, redness, bright, tex], -1).astype(np.float32)


def _mask(stem, size=WORK):
    return np.array(Image.open(MASK_DIR / f"{stem}.png").convert("L").resize((size, size), Image.NEAREST)) > 128


def _roi(stem, H, W):
    pp = PRIOR_DIR / f"{stem}.png"
    if not pp.exists():
        return np.ones((H, W), bool)
    prior = np.array(Image.open(pp).convert("L").resize((W, H), Image.NEAREST)) > 128
    k = max(3, int(NEAR_PRIOR_FRAC * max(H, W)))
    return cv2.dilate(prior.astype(np.uint8), np.ones((k, k), np.uint8)) > 0 if prior.any() else np.ones((H, W), bool)


def build_training(ids, rng):
    X, y = [], []
    for s in ids:
        rgb, nir = _load(s)
        F = features(rgb, nir).reshape(-1, 7)
        m = _mask(s).ravel()
        pos = np.where(m)[0]
        neg = np.where(~m)[0]
        for idxs in (pos, neg):
            take = idxs if len(idxs) <= N_PER_TILE else rng.choice(idxs, N_PER_TILE, replace=False)
            X.append(F[take]); y.append(m[take].astype(int))
    return np.concatenate(X), np.concatenate(y)


def train_clf(ids, rng):
    X, y = build_training(ids, rng)
    clf = RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=8,
                                 class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return clf


def predict_polys(clf, stem):
    """Return polygons in FULL-resolution pixel coords for one tile."""
    rgb, nir = _load(stem)
    H, W = rgb.shape[:2]
    prob = clf.predict_proba(features(rgb, nir).reshape(-1, 7))[:, 1].reshape(H, W)
    hot = (prob > PROB_THR) & _roi(stem, H, W)
    hot = binary_closing(binary_opening(hot, iterations=2), iterations=3)
    # scale factor back to the stored image size
    with Image.open(IMG_DIR / f"{stem}.png") as im:
        fullW, fullH = im.size
    sx, sy = fullW / W, fullH / H
    cnts, _ = cv2.findContours(hot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_BLOB_FRAC * H * W:
            continue
        ap = cv2.approxPolyDP(c, SIMPLIFY_FRAC * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) >= 3:
            polys.append([[float(x * sx), float(y * sy)] for x, y in ap])
    return polys, (fullW, fullH)


def loo_eval(labeled, rng, max_tiles=None):
    """Leave-one-out region IoU on the labeled tiles (quality-vs-#labels gauge). SUBSAMPLED to
    <=max_tiles held-out tiles for speed — full LOO trains one RandomForest PER tile (142 forests
    took ~9h). max_tiles<=0 skips the gauge entirely."""
    max_tiles = LOO_MAX_TILES if max_tiles is None else max_tiles
    if len(labeled) < 2 or max_tiles == 0:
        return None
    held_set = (labeled if len(labeled) <= max_tiles
                else [labeled[i] for i in rng.choice(len(labeled), max_tiles, replace=False)])
    ious = []
    for held in held_set:
        clf = train_clf([s for s in labeled if s != held], rng)
        rgb, nir = _load(held); H, W = rgb.shape[:2]
        prob = clf.predict_proba(features(rgb, nir).reshape(-1, 7))[:, 1].reshape(H, W)
        pred = binary_closing(binary_opening((prob > PROB_THR) & _roi(held, H, W), iterations=2), iterations=3)
        gt = _mask(held)
        if gt.sum() == 0:
            continue                       # no-damage tile: skip, don't credit a free 1.0
        inter = (pred & gt).sum(); union = (pred | gt).sum()
        ious.append(inter / union)         # union >= gt.sum() > 0, so safe
    return float(np.mean(ious)) if ious else None


def _reviewed(stem):
    """True only if you OPENED + SAVED this tile in Labelme (Labelme rewrites the version to its own
    on save). A tile still stamped at DRAFT_VERSION only carries a machine auto-draft = NOT reviewed,
    so it may be re-drafted. NOTE: do NOT also test _autodraft — Labelme 6.3.1 preserves that key
    through its `otherData` mechanism, so it survives a real save and would mark reviewed tiles as
    un-reviewed, letting a re-draft OVERWRITE your hand-labels (root-caused 2026-07-06)."""
    jp = LABEL_DIR / f"{stem}.json"
    if not jp.exists():
        return False
    try:
        d = json.loads(jp.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(d.get("version", "")) != DRAFT_VERSION


def main():
    labeled = sorted(p.stem for p in MASK_DIR.glob("*.png")) if MASK_DIR.exists() else []
    all_ids = sorted(p.stem for p in IMG_DIR.glob("*.png"))
    # (re)draft every tile that is NOT masked and NOT human-reviewed — i.e. tiles with no JSON OR only a
    # machine auto-draft. This lets Idea B UPGRADE the earlier prelabel_regions drafts as your labels
    # grow, WITHOUT deleting anything and WITHOUT ever overwriting a tile you reviewed.
    unlabeled = [s for s in all_ids
                 if s not in labeled and (OVERWRITE or not _reviewed(s))]
    print(f"labeled: {len(labeled)}  unlabeled to draft (absent or un-reviewed drafts): {len(unlabeled)}")
    if len(labeled) < 2:
        print("Need >=2 labeled tiles to train. Label a few by hand first "
              "(or use prelabel_regions.py for the very first ones).")
        return

    rng = np.random.default_rng(42)
    loo = loo_eval(labeled, rng)
    if loo is not None:
        print(f"leave-one-out region IoU (subsample of <={LOO_MAX_TILES} of your {len(labeled)} labels): "
              f"{loo:.2f} ({'usable' if loo > 0.4 else 'still rough — add more labels'})")

    clf = train_clf(labeled, rng)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in unlabeled:
        polys, (W, H) = predict_polys(clf, s)
        shapes = [{"label": "damage", "points": p, "group_id": None,
                   "shape_type": "polygon", "flags": {}} for p in polys]
        rel = os.path.relpath(IMG_DIR / f"{s}.png", LABEL_DIR).replace(os.sep, "/")
        (LABEL_DIR / f"{s}.json").write_text(json.dumps({
            "version": "5.5.0", "_autodraft": True, "flags": {}, "shapes": shapes,
            "imagePath": rel, "imageData": None, "imageHeight": H, "imageWidth": W,
        }, indent=2), encoding="utf-8")   # _autodraft => labelme_to_masks skips until you review+save
        n += 1
        print(f"  {s}: drafted {len(polys)} region(s)")
    print(f"\nDrafted {n} unlabeled tile(s). Open in Labelme and FIX the drafts.")


if __name__ == "__main__":
    main()
