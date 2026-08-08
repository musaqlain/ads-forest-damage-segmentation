# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Build 30cm seed tiles for self-labeling
#
# For each **2024 ADS damage polygon** this fetches a 30cm Oregon (OSIP 2024) image
# tile sized to contain that polygon, and rasterises the (misaligned) ADS polygon as
# a **prior** channel. Output = a folder of `(image, prior)` tiles + an index CSV,
# ready for you to redraw the tight damage boundary in `annotate_regions.py`.
#
# ## What we target and WHY (read this before running)
#
# The source geodatabase `DamageCombined` layer has **48,445** polygons for survey year
# 2024. They are very mixed, so we filter down to the *easiest, clearest* slice first
# (divide-and-conquer — batch 1). The filters below were chosen from inspecting the data:
#
# * **`DAMAGE_TYPE == "Mortality"`** — dead trees (92% of the data). This is the target.
#   Sick-tree classes (`Topkill`, `Crown Dieback`, `Defoliation …`) can be added later by
#   editing `DAMAGE_TYPES` — the code already supports them.
# * **`AREA_TYPE == "POLYGON"`** — a surveyor actually *drew* this boundary. The other half
#   of the rows have `AREA_TYPE` blank: those are single **point** observations that were
#   buffered into a perfect **circle** (their `Radius`/`BUFF_DIST` is set, their shape is
#   fake). A drawn polygon is a meaningful region hint; a buffer circle is not. Set
#   `AREA_TYPES = None` to also include the point-buffers (as pure location hints).
# * **Higher severity** (`PCT_AFFECTED` in Moderate/Severe/Very-Severe) — 74% of rows are
#   "Very Light (1-3%)", meaning only 1-3% of trees in the polygon are dead: nearly invisible
#   at 30cm and impossible to redraw. We start with the polygons where damage is actually
#   *visible*. Set `SEVERITY = None` to keep all severities.
# * **Oregon only** — OSIP imagery is Oregon-only, but the survey (USFS Region 6) also covers
#   Washington. ~63% of mortality polygons are in WA and would come back as blank white tiles.
#   We filter to an Oregon bounding box AND detect/skip blank tiles as a safety net.
# * **Size that fits one 30cm tile** — even the *smallest* real mortality polygon is ~244m
#   across; the median is over 1km. A polygon bigger than `MAX_WINDOW_M` can't be a single
#   30cm tile, so we skip it in this batch (those km-scale "envelopes" are a later, gridded
#   pass — see `TILE_MODE` note at the bottom). Batch 1 keeps the ~200–480m polygons.
#
# Runs **locally** (it just calls the public OSIP REST server) or on Colab. Set the paths
# and, if you want, the batch filters in the config cell.
#
# Output layout (matches what `annotate_regions.py` / `finetune_30cm.py` expect):
# ```
# OUT_DIR/
#   images/<id>.png     # RGB 30cm tile (size x size, true ~0.30 m/px)
#   priors/<id>.png     # rough ADS polygon rasterised (0/255) — the region hint
#   index.csv           # id, observation_id, damage_type, dca, host, pct_affected,
#                       #   area_m2, extent_m, window_m, size_px, gsd_m, lon, lat, bbox_6557
# ```

# %%
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from PIL import Image, ImageDraw

# %% [markdown]
# ## Config — SET THESE

# %%
# --- paths -------------------------------------------------------------------
# The 2024 unaligned ADS polygons (a File Geodatabase / .gdb).
POLY_PATH = r"D:\Opensource\Transform ADS polygons\data\INTERIM_R6_ADS_DATA2024.gdb"  # <-- CHANGE ME
# A .gdb is a CONTAINER of layers; the damage polygons live in the "DamageCombined" layer.
# (Setting this to the .gdb *filename* is what caused the "Layer could not be opened" error.)
POLY_LAYER = "DamageCombined"
OUT_DIR = Path("data/seed30cm")          # matches annotate_regions.py / finetune_30cm.py
APPEND_MODE = True                       # add a NEW batch to an existing OUT_DIR: skip any polygon
                                         #   already in index.csv, continue id numbering (0042, 0043, …),
                                         #   and APPEND to index.csv instead of overwriting it. So your
                                         #   already-annotated tiles are never touched or re-downloaded.
                                         #   Set False for a clean-slate build into an empty OUT_DIR.

# --- batch-1 filters (the "which polygons" decision) -------------------------
DAMAGE_TYPES = ["Mortality"]             # dead trees. Add "Topkill","Crown Dieback",... for sick trees. None = all.
AREA_TYPES   = ["POLYGON"]               # real drawn shapes only. None = also include buffered point observations.
SEVERITY     = ["Moderate (11-29%)",     # visible damage only. None = all severities.
                "Severe (30-50%)",
                "Very Severe (>50%)"]
# Oregon bounding box in lon/lat (OSIP imagery is Oregon-only). Roughly the whole state.
REGION_BBOX_4326 = (-124.7, 41.9, -116.4, 46.3)   # (lon_min, lat_min, lon_max, lat_max); None = no geo filter

# --- how many + which to prepare --------------------------------------------
MAX_TILES = 45               # this batch: ~40 NEW tiles to annotate (batch 1 used 42). Loop stops once
                             #   MAX_TILES are SAVED. Set to 3 first to sanity-check, then raise.
SAMPLE    = "diverse"        # "diverse" | "largest" | "random" | "first"
SEED      = 42

# --- tile geometry -----------------------------------------------------------
TARGET_GSD_M = 0.30          # metres per pixel we want in the output (true 30cm)
PAD_FRAC     = 0.25          # margin of healthy forest around the polygon on each side
                             #   (0.25 => polygon fills the middle ~65% of the tile, with green
                             #    context around it so the damage boundary is easy to see and
                             #    doesn't run off the tile edge). ~2x the old 0.12 margin.
MIN_WINDOW_M = 180.0         # smallest tile side in metres (context for tiny polygons)
MAX_WINDOW_M = 900.0         # was 700m for batch 1. Raised to 900m for batch 2 because the Moderate+
                             #   Oregon POLYGON pool at <=700m is EXHAUSTED (66 candidates, 42 used, ~24
                             #   left and most re-blank). <=900m yields ~100 fresh candidates while
                             #   staying high-severity (visible, annotatable). Trade-off: the biggest
                             #   polygons hit the 2048px cap, so their GSD drifts 0.34 -> 0.44 m/px
                             #   (still region-scale). Keep SEVERITY Moderate+ so damage stays traceable.
MIN_SIZE_PX  = 512           # clamp output pixel size
MAX_SIZE_PX  = 2048          # (OSIP is happy with PNG up to ~2048; keeps files annotatable)

# --- imagery server (public Oregon OSIP 2024, 30cm) --------------------------
OSIP_URL = ("https://imagery.oregonexplorer.info/arcgis/rest/services/"
            "OSIP_2024/OSIP_2024_WM/ImageServer")
OSIP_SR = 6557               # EPSG for the OSIP request (Oregon Lambert, metres) — used before, works
SAVE_NIR = True              # OSIP 2024 band 4 = near-infrared. Save it (nir/<id>.png) — NDVI is
                             #   the single best dead-tree signal + a useful model input channel.

# %% [markdown]
# ## Fetch helper (same pattern as demo_weak_augmentation.py) + blank-tile guard

# %%
def fetch_osip(bbox_6557, size):
    """bbox_6557 = (xmin, ymin, xmax, ymax) in EPSG:6557 -> (size,size,3) uint8 or None.

    Returns None on a server/JSON error OR when the tile is mostly nodata (white) — which
    is what OSIP returns outside its Oregon coverage.
    """
    xmin, ymin, xmax, ymax = bbox_6557
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": str(OSIP_SR), "imageSR": str(OSIP_SR),
        "size": f"{size},{size}", "format": "png", "pixelType": "U8",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    for attempt in range(3):
        try:
            r = requests.get(f"{OSIP_URL}/exportImage", params=params, timeout=120)
            r.raise_for_status()
            if "json" in r.headers.get("Content-Type", "") or r.content[:1] == b"{":
                print("  server returned JSON error"); return None
            arr = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"), np.uint8)
            # blank / nodata guard: OSIP paints uncovered area near-white (mean~253, std~0)
            near_white = (arr > 250).all(axis=2).mean()
            if arr.std() < 3 or near_white > 0.6:
                print(f"  blank/nodata tile (mean={arr.mean():.0f}, white={near_white:.0%}) — skipping")
                return None
            return arr
        except Exception as e:
            print(f"  attempt {attempt+1}/3 failed: {e}"); time.sleep(2 ** attempt)
    return None


def fetch_osip_nir(bbox_6557, size):
    """Fetch just the near-infrared band (OSIP band 3) as a (size,size) uint8 array, aligned
    to the same bbox/size as the RGB tile. Returns None on error (RGB tile is still kept)."""
    xmin, ymin, xmax, ymax = bbox_6557
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": str(OSIP_SR), "imageSR": str(OSIP_SR),
        "size": f"{size},{size}", "format": "png", "pixelType": "U8",
        "bandIds": "3",                                   # 0,1,2,3 = R,G,B,NIR
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    for attempt in range(3):
        try:
            r = requests.get(f"{OSIP_URL}/exportImage", params=params, timeout=120)
            r.raise_for_status()
            if "json" in r.headers.get("Content-Type", "") or r.content[:1] == b"{":
                return None
            return np.array(Image.open(io.BytesIO(r.content)).convert("L"), np.uint8)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def window_and_size(geom):
    """Choose a square window (m) that contains the polygon + padding, and the pixel size
    that keeps ~TARGET_GSD_M metres/pixel. Returns (window_m, size_px) or (None, None) if
    the polygon is too big for a single batch-1 tile."""
    minx, miny, maxx, maxy = geom.bounds
    extent_m = max(maxx - minx, maxy - miny)
    window_m = max(MIN_WINDOW_M, extent_m * (1.0 + 2.0 * PAD_FRAC))
    if window_m > MAX_WINDOW_M:
        return None, None
    size_px = int(round(window_m / TARGET_GSD_M))
    size_px = int(np.clip(size_px, MIN_SIZE_PX, MAX_SIZE_PX))
    return window_m, size_px


def square_bbox(geom, window_m):
    """Square bbox of side window_m centred on the polygon centroid (EPSG:6557)."""
    cx, cy = geom.centroid.x, geom.centroid.y
    half = window_m / 2.0
    return cx - half, cy - half, cx + half, cy + half


def rasterize(geom, bbox, size):
    """Rasterise a (Multi)Polygon into the tile pixel grid. Top-left = (xmin, ymax)."""
    xmin, ymin, xmax, ymax = bbox
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    for p in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
        xs, ys = p.exterior.xy
        px = [(x - xmin) / (xmax - xmin) * size for x in xs]
        py = [(ymax - y) / (ymax - ymin) * size for y in ys]   # north-up image
        d.polygon(list(zip(px, py)), fill=255)
    return np.array(im, np.uint8)

# %% [markdown]
# ## Load polygons, apply the batch-1 filters, pick a diverse subset

# %%
gdf = gpd.read_file(POLY_PATH, layer=POLY_LAYER) if POLY_LAYER else gpd.read_file(POLY_PATH)
gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
gdf = gdf.to_crs(epsg=OSIP_SR)                      # -> metres (equal-area) for area/tiles
gdf["area_m2"] = gdf.geometry.area
b = gdf.geometry.bounds
gdf["extent_m"] = np.maximum(b["maxx"] - b["minx"], b["maxy"] - b["miny"])
cen = gdf.geometry.centroid
lonlat = gpd.GeoSeries(cen, crs=gdf.crs).to_crs(4326)
gdf["lon"], gdf["lat"] = lonlat.x.values, lonlat.y.values
gdf["obs_id"] = gdf["OBSERVATION_ID"].astype(str).str.strip("{}")   # match index.csv's observation_id
print(f"loaded {len(gdf)} polygons from layer '{POLY_LAYER}'")


def _funnel(df, mask, label):
    kept = df[mask]
    print(f"  filter {label:<28s}: {len(kept):6d} kept (of {len(df)})")
    return kept


sel = gdf
if DAMAGE_TYPES is not None:
    sel = _funnel(sel, sel["DAMAGE_TYPE"].isin(DAMAGE_TYPES), f"DAMAGE_TYPE in {DAMAGE_TYPES}")
if AREA_TYPES is not None:
    sel = _funnel(sel, sel["AREA_TYPE"].isin(AREA_TYPES), f"AREA_TYPE in {AREA_TYPES}")
if SEVERITY is not None:
    sel = _funnel(sel, sel["PCT_AFFECTED"].isin(SEVERITY), "PCT_AFFECTED (visible damage)")
if REGION_BBOX_4326 is not None:
    lo_x, lo_y, hi_x, hi_y = REGION_BBOX_4326
    inbox = (sel["lon"].between(lo_x, hi_x)) & (sel["lat"].between(lo_y, hi_y))
    sel = _funnel(sel, inbox, "Oregon bbox (OSIP coverage)")
# size band: only polygons that fit a single ~30cm tile (window <= MAX_WINDOW_M)
fits = (sel["extent_m"] * (1.0 + 2.0 * PAD_FRAC)).clip(lower=MIN_WINDOW_M) <= MAX_WINDOW_M
sel = _funnel(sel, fits, f"fits one tile (window<= {MAX_WINDOW_M:.0f}m)")

# --- APPEND MODE: never re-download a polygon already in this OUT_DIR's index.csv ------------
start_idx = 0
if APPEND_MODE and (OUT_DIR / "index.csv").exists():
    prev = pd.read_csv(OUT_DIR / "index.csv")
    used = set(prev["observation_id"].astype(str))
    sel = _funnel(sel, ~sel["obs_id"].isin(used), f"exclude {len(used)} already-downloaded")
    # continue numbering after the highest existing tile so we never overwrite one
    existing = [int(p.stem) for p in (OUT_DIR / "images").glob("*.png") if p.stem.isdigit()]
    start_idx = (max(existing) + 1) if existing else len(prev)
    print(f"  APPEND_MODE on: new tiles numbered from {start_idx:04d}; index.csv will be appended")

if len(sel) == 0:
    raise SystemExit("No polygons pass the filters — relax DAMAGE_TYPES/AREA_TYPES/SEVERITY/size "
                     "(or, in APPEND_MODE, you may have already downloaded the whole pool).")


# severity order — most-severe first is the EASIEST to annotate (dense, obvious damage)
SEV_RANK = {"Very Severe (>50%)": 0, "Severe (30-50%)": 1, "Moderate (11-29%)": 2,
            "Light (4-10%)": 3, "Very Light (1-3%)": 4}


def order_diverse(df, seed=SEED):
    """Order the WHOLE pool by round-robin across DCA (damage-causal agent) so the batch
    spans many insects/diseases, while within each agent taking the EASIEST first
    (most severe, then smallest/most-compact). We order (not truncate) so the fetch loop
    can skip blank/oversize tiles and just take the next candidate until MAX_TILES saved."""
    df = df.copy()
    df["_sev"] = df["PCT_AFFECTED"].map(SEV_RANK).fillna(9)
    # easiest-first within each agent: most severe, then smallest area (compact patch)
    df = df.sort_values(["_sev", "area_m2"], kind="stable")
    groups = [grp for _, grp in df.groupby("DCA", sort=False)]
    picked, i = [], 0
    while any(len(g) for g in groups):
        g = groups[i % len(groups)]
        if len(g):
            picked.append(g.iloc[[0]])
            groups[i % len(groups)] = g.iloc[1:]
        i += 1
    return pd.concat(picked) if picked else df


# Order the full candidate pool; the loop below stops once MAX_TILES are SAVED.
if SAMPLE == "largest":
    ordered = sel.sort_values("area_m2", ascending=False)
elif SAMPLE == "random":
    ordered = sel.sample(frac=1.0, random_state=SEED)
elif SAMPLE == "diverse":
    ordered = order_diverse(sel)
else:  # "first"
    ordered = sel
ordered = ordered.reset_index(drop=True)
print(f"\n{len(ordered)} candidates pass filters; targeting {MAX_TILES} tiles ({SAMPLE}). "
      f"agents in pool: {dict(ordered['DCA'].value_counts())}")

# %% [markdown]
# ## Fetch + save each tile

# %%
(OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "priors").mkdir(parents=True, exist_ok=True)
if SAVE_NIR:
    (OUT_DIR / "nir").mkdir(parents=True, exist_ok=True)

rows = []
for _, row in ordered.iterrows():
    if len(rows) >= MAX_TILES:
        break
    geom = row.geometry
    window_m, size_px = window_and_size(geom)
    if window_m is None:
        print(f"  obs …{str(row.get('OBSERVATION_ID',''))[-6:]}: too big for one tile — skipping")
        continue
    bbox = square_bbox(geom, window_m)
    img = fetch_osip(bbox, size_px)
    if img is None:
        print(f"  obs …{str(row.get('OBSERVATION_ID',''))[-6:]}: fetch failed/blank, skipping")
        continue

    prior = rasterize(geom, bbox, size_px)
    tid = f"{start_idx + len(rows):04d}"   # continue ids after any existing tiles (APPEND_MODE)
    Image.fromarray(img).save(OUT_DIR / "images" / f"{tid}.png")
    Image.fromarray(prior).save(OUT_DIR / "priors" / f"{tid}.png")
    nir_ok = False
    if SAVE_NIR:
        nir = fetch_osip_nir(bbox, size_px)
        if nir is not None and nir.shape == img.shape[:2]:
            Image.fromarray(nir).save(OUT_DIR / "nir" / f"{tid}.png")
            nir_ok = True

    obs_id = str(row.get("OBSERVATION_ID", "")).strip("{}")
    rows.append({
        "id": tid,
        "observation_id": obs_id,
        "damage_type": row.get("DAMAGE_TYPE"),
        "dca": row.get("DCA"),
        "host": row.get("HOST"),
        "pct_affected": row.get("PCT_AFFECTED"),
        "area_m2": float(row["area_m2"]),
        "extent_m": float(row["extent_m"]),
        "window_m": round(window_m, 1),
        "size_px": int(size_px),
        "gsd_m": round(window_m / size_px, 4),
        "lon": round(float(row["lon"]), 6),
        "lat": round(float(row["lat"]), 6),
        "nir": nir_ok,
        "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
    })
    print(f"  saved {tid}: {size_px}px  {window_m:.0f}m  gsd={window_m/size_px:.3f}  "
          f"nir={'y' if nir_ok else 'n'}  {row.get('DCA')} / {row.get('PCT_AFFECTED')}")

new_df = pd.DataFrame(rows)
if APPEND_MODE and (OUT_DIR / "index.csv").exists():
    prev = pd.read_csv(OUT_DIR / "index.csv")
    out_df = pd.concat([prev, new_df], ignore_index=True)
    print(f"\nDone. {len(new_df)} NEW tiles appended ({len(out_df)} total) in {OUT_DIR}.")
else:
    out_df = new_df
    print(f"\nDone. {len(new_df)} tiles in {OUT_DIR}.")
out_df.to_csv(OUT_DIR / "index.csv", index=False)
print("Next: annotate the new tiles in Labelme, then run  python labelme_to_masks.py")

# %% [markdown]
# ## Later: the km-scale polygons (batch 2)
#
# ~90% of mortality polygons are larger than one 30cm tile (median > 1 km). Those are loose
# survey "envelopes" enclosing scattered mortality — you can't put one in a single tile.
# The way to use them later is a **gridded pass**: cover each big polygon with a grid of
# fixed 512px (=150m) tiles and keep only the tiles that intersect the polygon, then annotate
# the ones that actually show damage. That is Ben's "DeepForest patch tiling" idea. Keep this
# script for batch 1 (whole small polygons); add a grid loop when you get to the big ones.
