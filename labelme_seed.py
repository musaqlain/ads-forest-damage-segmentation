"""
labelme_seed.py  — (optional) pre-load the rough ADS polygon into Labelme as a guide
====================================================================================
Run this (in `ads_env`) BEFORE annotating if you want the rough ADS outline shown in
Labelme as a grey "prior" polygon you can trace or tighten. It vectorises each
`priors/<id>.png` and writes a Labelme `.json` per tile into `data/seed30cm/labelme/`.

The "prior" shapes are IGNORED by labelme_to_masks.py, so they never end up in your
training mask — they're purely a visual hint. Draw your OWN "damage" polygons on top
(or duplicate the prior shape in Labelme, relabel the copy to "damage", and tighten it).

NOTE: for auto-DRAFTS you'll actually fix (not just a guide), prefer `prelabel_learned.py`
(learns from your labels) or `prelabel_regions.py` (fixed threshold) — those write real
"damage" polygons. Use this only if you specifically want the ADS outline as a reference.

Run:
  python labelme_seed.py
  labelme data/seed30cm/images --output data/seed30cm/labelme --labels damage,prior

Skip this entirely if you'd rather draw from scratch — the converter works either way.
Set SEED30CM_DIR env var to point at a different dataset folder.
"""
import os
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

DATA = Path(os.environ.get("SEED30CM_DIR", "data/seed30cm"))
IMG_DIR = DATA / "images"
PRIOR_DIR = DATA / "priors"
LABEL_DIR = DATA / "labelme"
OVERWRITE = False          # True = replace existing JSONs (WARNING: discards annotations)


def prior_to_shapes(prior_path, min_area_px=80, simplify_frac=0.004):
    """Vectorise a 0/255 prior mask into simplified 'prior' polygon shapes."""
    m = (np.array(Image.open(prior_path).convert("L")) > 128).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        eps = simplify_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            shapes.append({
                "label": "prior",
                "points": approx.astype(float).tolist(),
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })
    return shapes


def main():
    if not IMG_DIR.exists():
        print(f"No images at {IMG_DIR}. Run build_30cm_seed_tiles.py first.")
        return
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in sorted(IMG_DIR.glob("*.png")):
        out = LABEL_DIR / f"{img.stem}.json"
        if out.exists() and not OVERWRITE:
            print(f"  {img.stem}: JSON exists, skipping (set OVERWRITE=True to replace)")
            continue

        prior = PRIOR_DIR / img.name
        shapes = prior_to_shapes(prior) if prior.exists() else []
        with Image.open(img) as im:
            W, H = im.size
        # imagePath is relative to the JSON's location (labelme/ -> ../images/<file>)
        rel = os.path.relpath(img, LABEL_DIR).replace(os.sep, "/")
        js = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": rel,
            "imageData": None,
            "imageHeight": H,
            "imageWidth": W,
        }
        out.write_text(json.dumps(js, indent=2), encoding="utf-8")
        n += 1
        print(f"  {img.stem}: seeded {len(shapes)} prior shape(s)")

    print(f"\nSeeded {n} Labelme JSON(s) in {LABEL_DIR}.")
    print("Now run:  labelme data/seed30cm/images --output data/seed30cm/labelme "
          "--labels damage,prior")


if __name__ == "__main__":
    main()
