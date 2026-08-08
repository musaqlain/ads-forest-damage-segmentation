"""
labelme_to_masks.py  — convert Labelme JSON annotations -> training masks
=========================================================================
Run this (in `ads_env`) AFTER you annotate the 30cm tiles in Labelme. It reads the
per-image Labelme `.json` files and rasterises your polygons into binary 0/255 masks,
pixel-aligned to each tile, saved as `data/seed30cm/masks/<id>.png` — exactly the format
`finetune_30cm.py` expects (same output as the old annotate_regions.py).

Labelme workflow (see the run report / PROJECT_MAP.md):
  labelme data/seed30cm/images --output data/seed30cm/labelme --labels damage
  # draw a polygon around each damage CLUSTER; label = "damage" (anything except "prior")
Then:
  python labelme_to_masks.py

Notes:
* Label "prior"/"hint" = the grey guide from labelme_seed.py; SKIPPED entirely.
* Label "ignore" (or "unsure"/"dontcare"/"skip") = a DON'T-CARE region: written to
  `data/seed30cm/ignore/<id>.png` and EXCLUDED from the training loss. Use it to fence off
  damage you can see but chose not to trace (so the model isn't taught those pixels are healthy).
* Every other polygon/rectangle/circle counts as damage.
* An image that has a JSON but no damage shapes -> empty mask (a valid "no-damage" example).
* An image with NO JSON -> not labeled (reported, no mask written).
* SAFETY GUARD: an auto-draft you NEVER opened+saved in Labelme (still stamped at DRAFT_VERSION
  by prelabel_*.py — Labelme bumps the version to its own on every save) is SKIPPED — otherwise
  un-vetted machine guesses would silently become training labels. Stale masks are removed.
  Set SKIP_UNREVIEWED=False to convert everything anyway.
* Set SEED30CM_DIR env var to point at a different dataset folder.
"""
import os
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATA = Path(os.environ.get("SEED30CM_DIR", "data/seed30cm"))
IMG_DIR = DATA / "images"
LABEL_DIR = DATA / "labelme"          # labelme --output dir (preferred)
MASK_DIR = DATA / "masks"
IGNORE_DIR = DATA / "ignore"          # optional "don't-care" mask (draw a shape labeled "ignore")
GUIDE_LABELS = {"prior", "hint", "_prior"}                 # grey guide (labelme_seed) — skipped entirely
IGNORE_LABELS = {"ignore", "unsure", "dontcare", "skip"}   # don't-care: not damage, not healthy

# --- reviewed-vs-draft guard -------------------------------------------------
# prelabel_regions.py / prelabel_learned.py stamp version=DRAFT_VERSION into every JSON they
# auto-write. When you OPEN a tile in Labelme and save, Labelme rewrites the file with ITS OWN
# version (e.g. 6.3.1). So "version still == DRAFT_VERSION" is the reliable "never reviewed" signal.
# NOTE: the drafters also add an _autodraft:true marker, but Labelme 6.3.1 PRESERVES unknown JSON
# keys through its `otherData` mechanism — _autodraft survives a real save, so it CANNOT be used to
# detect drafts (doing so wrongly skipped every reviewed tile, 2026-07-06). Version is the truth.
DRAFT_VERSION = "5.5.0"
AUTODRAFT_KEY = "_autodraft"   # kept for reference only; NOT used to gate (see note above)
SKIP_UNREVIEWED = True    # True = only convert tiles you actually opened + saved in Labelme


def shape_to_polygon(shape):
    """Labelme shape dict -> list of (x, y) vertices, or None if unusable."""
    st = shape.get("shape_type", "polygon")
    pts = shape.get("points", [])
    if st in ("polygon", "linestrip") and len(pts) >= 3:
        return [(float(x), float(y)) for x, y in pts]
    if st == "rectangle" and len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if st == "circle" and len(pts) == 2:
        (cx, cy), (ex, ey) = pts
        r = math.hypot(ex - cx, ey - cy)
        return [(cx + r * math.cos(t), cy + r * math.sin(t))
                for t in (i / 48 * 2 * math.pi for i in range(48))]
    return None


def image_size(stem):
    """(W, H) of the tile image for this id (authoritative for mask alignment)."""
    p = IMG_DIR / f"{stem}.png"
    if p.exists():
        with Image.open(p) as im:
            return im.size            # (W, H)
    return None


def json_to_mask(jpath):
    """Return (stem, damage_uint8, n_damage_shapes, ignore_uint8_or_None, is_unreviewed_draft)."""
    data = json.loads(Path(jpath).read_text(encoding="utf-8"))
    is_draft = str(data.get("version", "")) == DRAFT_VERSION   # see note above: version, not _autodraft
    stem = Path(data.get("imagePath", jpath.name)).stem or jpath.stem
    size = image_size(stem)
    if size is not None:
        W, H = size
    else:
        W, H = data.get("imageWidth"), data.get("imageHeight")
    im = Image.new("L", (W, H), 0)
    ig = Image.new("L", (W, H), 0)
    d, dig = ImageDraw.Draw(im), ImageDraw.Draw(ig)
    n = n_ig = 0
    for sh in data.get("shapes", []):
        lab = str(sh.get("label", "")).strip().lower()
        if lab in GUIDE_LABELS:
            continue                          # grey guide, not a label
        poly = shape_to_polygon(sh)
        if not poly:
            continue
        if lab in IGNORE_LABELS:
            dig.polygon(poly, fill=255); n_ig += 1     # don't-care region
        else:
            d.polygon(poly, fill=255); n += 1          # damage
    ignore = np.array(ig, np.uint8) if n_ig else None
    return stem, np.array(im, np.uint8), n, ignore, is_draft


def find_jsons():
    """stem -> json path; prefer the labelme/ output dir, fall back to images/."""
    found = {}
    for d in (LABEL_DIR, IMG_DIR):
        if d.exists():
            for j in sorted(d.glob("*.json")):
                found.setdefault(j.stem, j)
    return found


def main():
    if not IMG_DIR.exists():
        print(f"No images at {IMG_DIR}. Run build_30cm_seed_tiles.py first.")
        return
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    jsons = find_jsons()
    if not jsons:
        print(f"No Labelme .json files found in {LABEL_DIR} or {IMG_DIR}.")
        print("Annotate first:  labelme data/seed30cm/images --output data/seed30cm/labelme "
              "--labels damage")
        return

    written = 0
    skipped, removed = [], 0
    for stem, jpath in sorted(jsons.items()):
        try:
            stem, mask, n, ignore, is_draft = json_to_mask(jpath)
        except Exception as e:
            print(f"  {jpath.name}: FAILED ({e})")
            continue
        if SKIP_UNREVIEWED and is_draft:
            skipped.append(stem)
            # remove any stale mask/ignore left by a previous run that predated this guard
            for d in (MASK_DIR, IGNORE_DIR):
                f = d / f"{stem}.png"
                if f.exists():
                    f.unlink(); removed += 1
            continue
        Image.fromarray(mask).save(MASK_DIR / f"{stem}.png")
        written += 1
        px = int((mask > 0).sum())
        pct = 100.0 * px / mask.size if mask.size else 0.0
        tag = "empty (no-damage)" if n == 0 else f"{n} region(s)"
        ig_note = ""
        if ignore is not None:
            ignore[mask > 0] = 0                        # damage is always supervised; ignore only elsewhere
            IGNORE_DIR.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ignore).save(IGNORE_DIR / f"{stem}.png")
            ig_note = f"  +ignore {100.0 * (ignore > 0).mean():.1f}%"
        print(f"  {stem}: {tag:18s} {px:>9d} px  {pct:5.1f}% of tile{ig_note}")

    # report tiles that were never annotated
    all_ids = {p.stem for p in IMG_DIR.glob("*.png")}
    missing = sorted(all_ids - set(jsons))
    print(f"\nWrote {written} REVIEWED mask(s) to {MASK_DIR}.")
    if skipped:
        print(f"SKIPPED {len(skipped)} un-reviewed auto-draft(s) — never opened + saved in Labelme:")
        print(f"  {', '.join(skipped)}")
        if removed:
            print(f"  (removed {removed} stale mask/ignore file(s) from an earlier run)")
        print("  -> open these in Labelme and SAVE to include them (or set SKIP_UNREVIEWED=False).")
    if missing:
        print(f"{len(missing)} tile(s) have NO annotation yet: {', '.join(missing)}")


if __name__ == "__main__":
    main()
