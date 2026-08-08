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
# # Build a LARGE, geographically-diverse seed set (positives + confirmed negatives)
#
# This is the "80% = grow the RIGHT data" push (Josh's post-mid-term guidance). It is a
# superset of `build_30cm_seed_tiles.py` with two upgrades that directly answer the mentor:
#
# 1. **Geographic balancing (the whole point).** The 2024 Mortality/POLYGON/Oregon pool is
#    lopsided — ~52% of Moderate+ candidates sit in ONE central-Cascades Douglas-fir-beetle
#    outbreak. Selecting "diverse by damage-agent" (the old script) still lands mostly in one
#    place. Here we KMeans-cluster candidate centroids into `N_AREAS` spatial clusters and
#    **round-robin across clusters** (small clusters first, easiest-tile first within each),
#    so the batch spreads across ≥5 distinct areas. This is what makes spatial cross-validation
#    trustworthy and whole-area holdout a real generalization test (fixes the [42,4,2,7,2] folds).
#
# 2. **Confirmed negatives (Josh's specific ask).** Not random empty tiles — tiles INSIDE the
#    `SURVEYED_AREAS` footprint (a surveyor looked) with NO damage of ANY type nearby (they
#    found nothing = a trustworthy "no dead trees here"). These teach the model to stay quiet
#    where there is genuinely no damage, WITHOUT a polygon prior (deployment-like). We also
#    subtract `NOT_FLOWN` and require real OSIP imagery (blank-guard).
#
# Output = the SAME `data/seed30cm/` layout the rest of the pipeline expects, with a new
# `role` column in `index.csv` ("damage" vs "negative"). Positives await annotation in Labelme;
# negatives are confirmed-empty by construction, so we write their all-zero mask immediately.
#
# ```
# OUT_DIR/
#   images/<id>.png   priors/<id>.png   nir/<id>.png     # all tiles
#   masks/<id>.png                                       # NEGATIVES only (all-zero) — positives get theirs from Labelme
#   index.csv         # ... + role ("damage"|"negative")
# ```
#
# Runs locally or on Colab (public OSIP REST server). Fetch is network-bound: ~500 tiles ≈ 20–40 min.

# %%
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from PIL import Image, ImageDraw
from shapely.geometry import box, Point
from shapely.prepared import prep

warnings.filterwarnings("ignore")
# Windows consoles default to cp1252 and choke on any non-ASCII in a print(). Force UTF-8
# so progress logs never crash the run mid-download.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# %% [markdown]
# ## Config — SET THESE

# %%
# --- paths -------------------------------------------------------------------
POLY_PATH = r"D:\Opensource\Transform ADS polygons\data\INTERIM_R6_ADS_DATA2024.gdb"
DAMAGE_LAYER   = "DamageCombined"     # all 2024 damage features (48,445)
SURVEY_LAYER   = "SURVEYED_AREAS"     # the flown/surveyed footprint (1 MultiPolygon) — basis for negatives
NOTFLOWN_LAYER = "NOT_FLOWN"          # unflown holes to subtract from the negative sampling region
OUT_DIR = Path("data/seed30cm")       # same dir the rest of the pipeline reads
APPEND_MODE = True                    # skip anything already in index.csv; continue id numbering; append index.csv

# --- how many + geographic spread -------------------------------------------
MAX_POS   = 350        # positive (damage) tiles to SAVE this run
MAX_NEG   = 150        # confirmed-negative tiles to SAVE this run  (≈30% negatives → 500 total)
N_AREAS   = 10         # spatial clusters to balance positives across (≥5; ~10 matches the ~8–10 real outbreaks)
SEED      = 42

# --- positive-pool filters (visible, drawable Mortality regions) ------------
DAMAGE_TYPES = ["Mortality"]                 # dead trees (the target). None = all damage types.
AREA_TYPES   = ["POLYGON"]                    # surveyor-drawn shapes only (not fake point-buffer circles)
# Severities kept. PCT_AFFECTED = DENSITY of dead trees inside the polygon, NOT its size. Moderate+
# (>=11% dead) reads as an obvious brown patch you can trace; Light (4-10%) is often a BIG polygon
# (median extent ~465 m) holding only a sparse scatter of dead trees -> you have to zoom to hunt for
# them -> slow, noisy annotation. Dropped Light: verified on the 2024 gdb the Moderate+ pool is 1047
# tiles across ALL 10 spatial clusters, so we keep full geographic spread (need only MAX_POS=350).
SEVERITY     = ["Very Severe (>50%)", "Severe (30-50%)", "Moderate (11-29%)"]
REGION_BBOX_4326 = (-124.7, 41.9, -116.4, 46.3)   # Oregon (OSIP is Oregon-only); None = no geo filter

# --- tile geometry (positives) ----------------------------------------------
TARGET_GSD_M = 0.30    # metres/pixel we want (true 30cm)
PAD_FRAC     = 0.25    # green margin of healthy forest around the polygon
MIN_WINDOW_M = 180.0
MAX_WINDOW_M = 1500.0  # TRUE metres now (see OSIP_SR note). Generous so the geographic round-robin can
                       #   reach thin clusters; the Moderate+ Oregon pool that fits one tile is well >500,
                       #   plenty for MAX_POS. The biggest polygons hit the 2048px cap so their GSD drifts
                       #   up to ~0.73 m/px — still region-scale; `gsd_m` is recorded per tile so you can
                       #   filter to ≤0.45 later if you want pixel purity. Lower this for tighter GSD.
MIN_SIZE_PX  = 512
MAX_SIZE_PX  = 2048

# --- confirmed negatives ----------------------------------------------------
NEG_WINDOW_M   = 300.0     # fixed tile side for negatives (~1000px @ 0.30 m/px)
NEG_DAMAGE_BUF = 150.0     # keep a negative tile-center at least this far (m) from ANY damage feature
NEG_MIN_SPACING = 800.0    # minimum spacing (m) between accepted negatives (spread them out)
NEG_OVERSAMPLE = 60000     # random candidate centers to draw inside surveyed-Oregon before filtering

# --- imagery server (public Oregon OSIP 2024, 30cm, 4-band) -----------------
OSIP_URL = ("https://imagery.oregonexplorer.info/arcgis/rest/services/"
            "OSIP_2024/OSIP_2024_WM/ImageServer")
OSIP_SR = 6556         # EPSG for OSIP requests + ALL metric geometry. 6556 = "NAD83(2011)/Oregon GIC
                       #   Lambert (metre)". Do NOT use 6557 — its twin is in FEET, so treating it as
                       #   metres silently made every window/buffer/spacing/GSD 0.3048x wrong (a latent
                       #   bug inherited from build_30cm_seed_tiles.py). OSIP serves 6556 identically to
                       #   6557 (same imagery, verified), so this one change makes every _m value truthful.
SAVE_NIR = True        # OSIP band 3 (0-indexed) = near-infrared → nir/<id>.png (best dead-tree signal)

# %% [markdown]
# ## Fetch helpers (proven pattern from build_30cm_seed_tiles.py) + blank guard

# %%
def fetch_osip(bbox_6557, size, bandIds=None):
    """bbox in EPSG:6557 -> (size,size,3) uint8, or None on error / blank(nodata) tile.
    bandIds=None -> RGB; bandIds='3' -> the single NIR band (returned as (size,size))."""
    xmin, ymin, xmax, ymax = bbox_6557
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": str(OSIP_SR), "imageSR": str(OSIP_SR),
        "size": f"{size},{size}", "format": "png", "pixelType": "U8",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    if bandIds is not None:
        params["bandIds"] = bandIds
    for attempt in range(3):
        try:
            r = requests.get(f"{OSIP_URL}/exportImage", params=params, timeout=120)
            r.raise_for_status()
            if "json" in r.headers.get("Content-Type", "") or r.content[:1] == b"{":
                return None
            if bandIds is not None:                       # NIR: single band, no blank-guard (RGB already guarded)
                return np.array(Image.open(io.BytesIO(r.content)).convert("L"), np.uint8)
            arr = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"), np.uint8)
            near_white = (arr > 250).all(axis=2).mean()   # OSIP paints uncovered area near-white
            if arr.std() < 3 or near_white > 0.6:
                return None                                # blank / outside coverage
            return arr
        except Exception:
            time.sleep(2 ** attempt)
    return None


def window_and_size(geom):
    minx, miny, maxx, maxy = geom.bounds
    extent_m = max(maxx - minx, maxy - miny)
    window_m = max(MIN_WINDOW_M, extent_m * (1.0 + 2.0 * PAD_FRAC))
    if window_m > MAX_WINDOW_M:
        return None, None
    size_px = int(np.clip(round(window_m / TARGET_GSD_M), MIN_SIZE_PX, MAX_SIZE_PX))
    return window_m, size_px


def square_bbox_centre(cx, cy, window_m):
    half = window_m / 2.0
    return cx - half, cy - half, cx + half, cy + half


def rasterize(geom, bbox, size):
    xmin, ymin, xmax, ymax = bbox
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    for p in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
        xs, ys = p.exterior.xy
        px = [(x - xmin) / (xmax - xmin) * size for x in xs]
        py = [(ymax - y) / (ymax - ymin) * size for y in ys]     # north-up
        d.polygon(list(zip(px, py)), fill=255)
    return np.array(im, np.uint8)

# %% [markdown]
# ## Load damage polygons, filter the positive pool

# %%
SEV_RANK = {"Very Severe (>50%)": 0, "Severe (30-50%)": 1, "Moderate (11-29%)": 2,
            "Light (4-10%)": 3, "Very Light (1-3%)": 4}

gdf = gpd.read_file(POLY_PATH, layer=DAMAGE_LAYER)
gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs(epsg=OSIP_SR).copy()
gdf["area_m2"] = gdf.geometry.area
b = gdf.geometry.bounds
gdf["extent_m"] = np.maximum(b["maxx"] - b["minx"], b["maxy"] - b["miny"])
cen = gdf.geometry.centroid
gdf["cx"], gdf["cy"] = cen.x.values, cen.y.values
ll = gpd.GeoSeries(cen, crs=gdf.crs).to_crs(4326)
gdf["lon"], gdf["lat"] = ll.x.values, ll.y.values
gdf["obs_id"] = gdf["OBSERVATION_ID"].astype(str).str.strip("{}")
print(f"loaded {len(gdf)} damage features from '{DAMAGE_LAYER}'")


def _funnel(df, mask, label):
    kept = df[mask]
    print(f"  filter {label:<30s}: {len(kept):6d} kept (of {len(df)})")
    return kept


sel = gdf
if DAMAGE_TYPES is not None:
    sel = _funnel(sel, sel["DAMAGE_TYPE"].isin(DAMAGE_TYPES), f"DAMAGE_TYPE in {DAMAGE_TYPES}")
if AREA_TYPES is not None:
    sel = _funnel(sel, sel["AREA_TYPE"].isin(AREA_TYPES), f"AREA_TYPE in {AREA_TYPES}")
if SEVERITY is not None:
    sel = _funnel(sel, sel["PCT_AFFECTED"].isin(SEVERITY), "PCT_AFFECTED (visible)")
if REGION_BBOX_4326 is not None:
    lo_x, lo_y, hi_x, hi_y = REGION_BBOX_4326
    sel = _funnel(sel, sel["lon"].between(lo_x, hi_x) & sel["lat"].between(lo_y, hi_y), "Oregon bbox")
fits = (sel["extent_m"] * (1.0 + 2.0 * PAD_FRAC)).clip(lower=MIN_WINDOW_M) <= MAX_WINDOW_M
sel = _funnel(sel, fits, f"fits one tile (<={MAX_WINDOW_M:.0f}m)")

# APPEND: never re-download a polygon already in index.csv; continue id numbering.
start_idx, prev_index = 0, None
already_obs = set()
if APPEND_MODE and (OUT_DIR / "index.csv").exists():
    prev_index = pd.read_csv(OUT_DIR / "index.csv")
    already_obs = set(prev_index["observation_id"].astype(str))
    sel = _funnel(sel, ~sel["obs_id"].isin(already_obs), f"exclude {len(already_obs)} already-downloaded")
    existing = [int(p.stem) for p in (OUT_DIR / "images").glob("*.png") if p.stem.isdigit()]
    start_idx = (max(existing) + 1) if existing else len(prev_index)
    print(f"  APPEND_MODE: new ids start at {start_idx:04d}")

if len(sel) == 0:
    raise SystemExit("No positive polygons pass the filters (or the pool is exhausted in APPEND_MODE).")

# %% [markdown]
# ## Geographic balancing: KMeans clusters -> round-robin (small clusters first, easiest tile first)

# %%
from sklearn.cluster import KMeans

k_areas = int(min(N_AREAS, len(sel)))
sel = sel.copy()
sel["_area"] = KMeans(n_clusters=k_areas, n_init=10, random_state=SEED).fit_predict(sel[["cx", "cy"]].values)
sel["_sev"] = sel["PCT_AFFECTED"].map(SEV_RANK).fillna(9)

# report the spatial spread we're balancing over
print(f"\n{len(sel)} candidate positives across {k_areas} spatial clusters:")
for a, g in sel.groupby("_area"):
    print(f"  area {a}: n={len(g):>3}  centre=({g['lon'].mean():.2f},{g['lat'].mean():.2f})  "
          f"top DCA={g['DCA'].value_counts().index[0] if g['DCA'].notna().any() else '?'}")


def _dca_interleave(grp):
    """Within one cluster: easiest-first (severe, then compact), then round-robin across DCA agents."""
    grp = grp.sort_values(["_sev", "area_m2"], kind="stable")
    dgroups = [g for _, g in grp.groupby("DCA", sort=False, dropna=False)]   # dropna=False: never silently drop null-DCA rows
    out, i = [], 0
    while any(len(g) for g in dgroups):
        g = dgroups[i % len(dgroups)]
        if len(g):
            out.append(g.iloc[[0]]); dgroups[i % len(dgroups)] = g.iloc[1:]
        i += 1
    return pd.concat(out) if out else grp


area_queues = {a: _dca_interleave(g).reset_index(drop=True) for a, g in sel.groupby("_area")}
# round-robin across areas (SMALL clusters first so they're never starved); pure round-robin already
# caps how much the giant Cascades cluster can dominate, because the loop stops once MAX_POS are saved.
order_areas = sorted(area_queues, key=lambda a: len(area_queues[a]))
picked, ptr, remaining = [], {a: 0 for a in area_queues}, True
while remaining:
    remaining = False
    for a in order_areas:
        q = area_queues[a]
        if ptr[a] < len(q):
            picked.append(q.iloc[[ptr[a]]]); ptr[a] += 1; remaining = True
ordered_pos = pd.concat(picked).reset_index(drop=True)
print(f"\npositive fetch order built ({len(ordered_pos)} candidates; loop stops at {MAX_POS} saved)")

# %% [markdown]
# ## Fetch + save POSITIVE tiles (image + rough ADS prior + NIR)

# %%
for sub in ("images", "priors", "masks"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
if SAVE_NIR:
    (OUT_DIR / "nir").mkdir(parents=True, exist_ok=True)

rows = []
for _, row in ordered_pos.iterrows():
    if len(rows) >= MAX_POS:
        break
    window_m, size_px = window_and_size(row.geometry)
    if window_m is None:
        continue
    bbox = square_bbox_centre(row["cx"], row["cy"], window_m)
    img = fetch_osip(bbox, size_px)
    if img is None:
        continue
    prior = rasterize(row.geometry, bbox, size_px)
    tid = f"{start_idx + len(rows):04d}"
    Image.fromarray(img).save(OUT_DIR / "images" / f"{tid}.png")
    Image.fromarray(prior).save(OUT_DIR / "priors" / f"{tid}.png")
    nir_ok = False
    if SAVE_NIR:
        nir = fetch_osip(bbox, size_px, bandIds="3")
        if nir is not None and nir.shape == img.shape[:2]:
            Image.fromarray(nir).save(OUT_DIR / "nir" / f"{tid}.png"); nir_ok = True
    rows.append({
        "id": tid, "role": "damage",
        "observation_id": row["obs_id"], "damage_type": row.get("DAMAGE_TYPE"),
        "dca": row.get("DCA"), "host": row.get("HOST"), "pct_affected": row.get("PCT_AFFECTED"),
        "area_m2": float(row["area_m2"]), "extent_m": float(row["extent_m"]),
        "window_m": round(window_m, 1), "size_px": int(size_px), "gsd_m": round(window_m / size_px, 4),
        "lon": round(float(row["lon"]), 6), "lat": round(float(row["lat"]), 6),
        "area_cluster": int(row["_area"]), "nir": nir_ok,
        "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
    })
    if len(rows) % 10 == 0 or len(rows) <= 3:
        print(f"  [pos {len(rows):>3}/{MAX_POS}] {tid} area{int(row['_area'])} "
              f"{row.get('DCA')} / {row.get('PCT_AFFECTED')}  gsd={window_m/size_px:.2f}")
print(f"saved {len(rows)} positive tiles")
pos_rows = rows

# %% [markdown]
# ## Confirmed NEGATIVES: inside SURVEYED_AREAS ∩ Oregon, outside NOT_FLOWN, far from ANY damage
#
# A confirmed negative = a place a surveyor covered and reported NO damage. We sample tile centers
# inside the surveyed footprint (minus unflown holes), require them to be ≥`NEG_DAMAGE_BUF` from every
# damage feature (any type, including point observations), spread them out, and keep only those with
# real OSIP imagery. Prior channel is all-zero (deployment-like: no polygon), and because they are
# empty by construction we write their all-zero mask now.

# %%
neg_rows = []
if MAX_NEG > 0:
    surv = gpd.read_file(POLY_PATH, layer=SURVEY_LAYER).to_crs(OSIP_SR)
    notflown = gpd.read_file(POLY_PATH, layer=NOTFLOWN_LAYER).to_crs(OSIP_SR)
    orbox_6557 = gpd.GeoSeries([box(*REGION_BBOX_4326)], crs=4326).to_crs(OSIP_SR).iloc[0]
    region = surv.geometry.union_all().intersection(orbox_6557).difference(notflown.geometry.union_all())
    region_prep = prep(region)
    minx, miny, maxx, maxy = region.bounds

    # all Oregon-area damage geometry to avoid (buffer so negatives clear the damage by NEG_DAMAGE_BUF)
    dmg = gdf[gdf["lon"].between(REGION_BBOX_4326[0], REGION_BBOX_4326[2]) &
              gdf["lat"].between(REGION_BBOX_4326[1], REGION_BBOX_4326[3])]
    dmg_idx = dmg.sindex
    print(f"negative region bounds {region.bounds}; avoiding {len(dmg)} Oregon damage features")

    # offset the RNG by start_idx so a SECOND APPEND run draws DIFFERENT centers — the fixed SEED alone
    # would reproduce the identical stream and re-download the same negatives as brand-new ids.
    rng = np.random.default_rng(SEED + start_idx)
    cxs = rng.uniform(minx, maxx, NEG_OVERSAMPLE)
    cys = rng.uniform(miny, maxy, NEG_OVERSAMPLE)
    pos_centres = np.array([[(r["xmin"] + r["xmax"]) / 2,
                             (r["ymin"] + r["ymax"]) / 2] for r in pos_rows]) if pos_rows else np.empty((0, 2))
    # pre-seed `accepted` with negatives saved by earlier runs (from index.csv) so new negatives keep
    # NEG_MIN_SPACING away from them and we never place two negatives on the same spot across runs.
    accepted = []            # accepted (cx,cy) for spacing checks
    if prev_index is not None and "role" in prev_index.columns:
        _pn = prev_index[(prev_index["role"] == "negative") & prev_index["xmin"].notna()]
        accepted += [((r["xmin"] + r["xmax"]) / 2.0, (r["ymin"] + r["ymax"]) / 2.0)
                     for _, r in _pn.iterrows()]
        if accepted:
            print(f"  carrying {len(accepted)} prior negatives forward for spacing/dedup")
    half = NEG_WINDOW_M / 2.0
    for cx, cy in zip(cxs, cys):
        if len(neg_rows) >= MAX_NEG:
            break
        # the WHOLE tile footprint must sit inside surveyed ∩ Oregon − NOT_FLOWN, not just the center —
        # else a boundary tile spills into unflown ground where real damage would be UNRECORDED, and an
        # all-zero "confirmed negative" mask there would teach the model to suppress genuine mortality.
        if not region_prep.contains(Point(cx, cy)):
            continue                                       # cheap center reject first (kills most candidates)
        if not region_prep.contains(box(cx - half, cy - half, cx + half, cy + half)):
            continue                                       # then the strict full-footprint containment
        # far from any damage: query damage sindex within (tile half + buffer)
        q = box(cx - half - NEG_DAMAGE_BUF, cy - half - NEG_DAMAGE_BUF,
                cx + half + NEG_DAMAGE_BUF, cy + half + NEG_DAMAGE_BUF)
        cand = list(dmg_idx.intersection(q.bounds))
        if cand and dmg.iloc[cand].geometry.intersects(q).any():
            continue
        # spacing vs already-accepted negatives (and vs positive tiles)
        if accepted:
            d = np.hypot(np.array(accepted)[:, 0] - cx, np.array(accepted)[:, 1] - cy)
            if d.min() < NEG_MIN_SPACING:
                continue
        if len(pos_centres) and np.hypot(pos_centres[:, 0] - cx, pos_centres[:, 1] - cy).min() < NEG_MIN_SPACING:
            continue
        bbox = square_bbox_centre(cx, cy, NEG_WINDOW_M)
        size_px = int(np.clip(round(NEG_WINDOW_M / TARGET_GSD_M), MIN_SIZE_PX, MAX_SIZE_PX))
        img = fetch_osip(bbox, size_px)
        if img is None:                                   # blank/no OSIP coverage here — skip
            continue
        accepted.append((cx, cy))
        tid = f"{start_idx + len(pos_rows) + len(neg_rows):04d}"
        Image.fromarray(img).save(OUT_DIR / "images" / f"{tid}.png")
        Image.fromarray(np.zeros((size_px, size_px), np.uint8)).save(OUT_DIR / "priors" / f"{tid}.png")
        Image.fromarray(np.zeros((size_px, size_px), np.uint8)).save(OUT_DIR / "masks" / f"{tid}.png")
        nir_ok = False
        if SAVE_NIR:
            nir = fetch_osip(bbox, size_px, bandIds="3")
            if nir is not None and nir.shape == img.shape[:2]:
                Image.fromarray(nir).save(OUT_DIR / "nir" / f"{tid}.png"); nir_ok = True
        lon, lat = gpd.GeoSeries([box(*bbox).centroid], crs=OSIP_SR).to_crs(4326).iloc[0].coords[0]
        neg_rows.append({
            "id": tid, "role": "negative", "observation_id": "", "damage_type": "",
            "dca": "", "host": "", "pct_affected": "", "area_m2": 0.0, "extent_m": 0.0,
            "window_m": NEG_WINDOW_M, "size_px": size_px, "gsd_m": round(NEG_WINDOW_M / size_px, 4),
            "lon": round(lon, 6), "lat": round(lat, 6), "area_cluster": -1, "nir": nir_ok,
            "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
        })
        if len(neg_rows) % 10 == 0 or len(neg_rows) <= 3:
            print(f"  [neg {len(neg_rows):>3}/{MAX_NEG}] {tid} ({lon:.2f},{lat:.2f})")
    print(f"saved {len(neg_rows)} confirmed-negative tiles")

# %% [markdown]
# ## Write / append index.csv

# %%
new_df = pd.DataFrame(pos_rows + neg_rows)
if APPEND_MODE and prev_index is not None:
    if "role" not in prev_index.columns:
        prev_index["role"] = "damage"                     # older tiles predate the role column
    out_df = pd.concat([prev_index, new_df], ignore_index=True)
    print(f"\nDone. {len(new_df)} NEW tiles ({len(pos_rows)} damage + {len(neg_rows)} negative) "
          f"appended -> {len(out_df)} total in {OUT_DIR}.")
else:
    out_df = new_df
    print(f"\nDone. {len(new_df)} tiles ({len(pos_rows)} damage + {len(neg_rows)} negative) in {OUT_DIR}.")
out_df.to_csv(OUT_DIR / "index.csv", index=False)

# spread report — this is the number that matters (are we across ≥5 areas, balanced?)
if pos_rows:
    pc = pd.DataFrame(pos_rows)["area_cluster"].value_counts().sort_index()
    print("positives per spatial area:", dict(pc))
print("\nNext: annotate the new POSITIVE tiles in Labelme (negatives already have zero masks),")
print("then run  python labelme_to_masks.py  and rebuild manifest with  python dataset_manifest.py")
