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
# # Day 1 — Diagnose the 38 real Monica pairs
#
# **Goal:** answer two questions *with numbers* before we build any model:
#
# 1. **Is the correction a reshape, not just a shift?**  If yes, an affine
#    (shift/rotate/scale) model can't reproduce it -> we need segmentation.
#    We test this by removing the best possible shift (and scale) and checking
#    whether the polygons still don't overlap.
# 2. **Is the damage even visible in the NAIP imagery?**  If the inside of
#    Monica's green polygon looks the same as just outside it, then *no* model
#    can learn it from these pixels, and we'd need better imagery. We test this
#    by comparing a greenness/vigour index inside green vs. just outside.
#
# Run **Part A** first (instant, geometry only). Run **Part B** if Part A says
# "reshape" (it needs Earth Engine + a few minutes).

# %% [markdown]
# ## Setup (same auth/paths as colab_explore_real_pairs.py)

# %%
import io
import urllib.request

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shapely import affinity
from shapely.ops import unary_union
from PIL import Image, ImageDraw

from google.colab import drive, auth
try:
    drive.mount('/content/drive')
except Exception:
    pass

GDB_PATH = ('/content/drive/MyDrive/Data/Historic_ADS_Data_Correct_GSC/'
            'Historic_ADS_Data_Correct_GSC/Data/OR_ADS_NAIP_Transform_2009.gdb')

# %% [markdown]
# ## Load + pair (Original = red/misaligned, Transform_1 'Reshape' = green/GT)

# %%
gdf_original = gpd.read_file(GDB_PATH, layer='OR_ADS_DAMAGE_AREA_R6_2009_Original')
gdf_corrected = gpd.read_file(GDB_PATH, layer='OR_ADS_DAMAGE_AREA_R6_2009_Transfrom_1')

orig = gdf_original[gdf_original['Checked'].isin(['Reshape', 'Resahpe'])].copy()
corr = gdf_corrected[gdf_corrected['Transform'] == 'Reshape'].copy()

paired = orig[['DAMAGE_AREA_ID', 'geometry', 'DCA_COMMON_NAME']].merge(
    corr[['DAMAGE_AREA_ID', 'geometry']],
    on='DAMAGE_AREA_ID', suffixes=('_orig', '_corr'))
print(f"Paired reshape polygons: {len(paired)}")
PROJ_CRS = gdf_original.crs   # projected, units = metres (used for area/shift)

# %% [markdown]
# ## PART A — Geometry diagnosis (no imagery, instant)
#
# For each pair we compute, in metres:
# - **shift** = distance between centroids
# - **area ratio** = green area / red area (how much smaller is the redraw?)
# - **IoU(raw)** = overlap of red & green as-is (1 = identical, 0 = disjoint)
# - **IoU(after best shift)** = overlap after sliding red so its centre sits on
#   green's centre. *If this is still low, translation can't fix it.*
# - **IoU(after shift+scale)** = also rescale red to green's area first.
#   *If THIS is still low, no affine can fix it -> reshape confirmed.*

# %%
def iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0

rows = []
for _, r in paired.iterrows():
    o, c = r['geometry_orig'], r['geometry_corr']
    oc, cc = o.centroid, c.centroid
    shift = float(np.hypot(oc.x - cc.x, oc.y - cc.y))

    # remove translation: move red centroid onto green centroid
    o_shift = affinity.translate(o, xoff=cc.x - oc.x, yoff=cc.y - oc.y)
    # also remove scale: rescale red to match green's area, about its centroid
    s = float(np.sqrt(c.area / o.area)) if o.area > 0 else 1.0
    o_shift_scale = affinity.scale(o_shift, xfact=s, yfact=s, origin=cc)

    rows.append(dict(
        id=r['DAMAGE_AREA_ID'],
        dca=r.get('DCA_COMMON_NAME', '?'),
        shift_m=shift,
        area_ratio=float(c.area / o.area) if o.area > 0 else np.nan,
        iou_raw=iou(o, c),
        iou_best_shift=iou(o_shift, c),
        iou_best_shift_scale=iou(o_shift_scale, c),
    ))

df = pd.DataFrame(rows)
pd.set_option('display.float_format', lambda v: f"{v:.3f}")
print(df.to_string(index=False))

print("\n--- SUMMARY (medians across 38 pairs) ---")
print(f"  shift                : {df.shift_m.median():.0f} m")
print(f"  area ratio green/red : {df.area_ratio.median():.2f}  "
      f"(<<1 means green is much smaller)")
print(f"  IoU raw              : {df.iou_raw.median():.2f}")
print(f"  IoU after best shift : {df.iou_best_shift.median():.2f}")
print(f"  IoU after shift+scale: {df.iou_best_shift_scale.median():.2f}")
print("\n  Interpretation: if 'IoU after shift+scale' is still low (say < 0.5),")
print("  then NO affine transform can turn red into green -> it is a genuine")
print("  RESHAPE -> segmentation is the right tool.")
df.to_csv('/content/drive/MyDrive/Data/day1_geometry_diagnosis.csv', index=False)

# %%
# Quick visual: how much does each affine 'upgrade' help the overlap?
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([df.iou_raw, df.iou_best_shift, df.iou_best_shift_scale],
           labels=['raw', '+best shift', '+best shift&scale'])
ax.set_ylabel('IoU (overlap) with Monica\'s green')
ax.set_title('Can an affine transform explain the correction?\n'
             '(bars staying low = no -> reshape -> segmentation)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Data/day1_affine_cannot_explain.png', dpi=150)
plt.show()

# %% [markdown]
# ## PART B — Is the damage visible in the imagery?
#
# We fetch 2009 NAIP for each pair, rasterise the red & green polygons into the
# image, and compare a **greenness index** (ExG = 2G - R - B; lower = browner =
# more likely dead/stressed) in three zones:
#   - inside green (Monica's damage)
#   - red-but-outside-green (area she rejected)
#   - background (outside both)
#
# If "inside green" is clearly **browner** than its surroundings, the damage is
# visible and a model can learn it. If all three are similar, the signal isn't in
# these pixels (we'd need higher-res / different imagery).

# %%
import ee
try:
    auth.authenticate_user()
    ee.Initialize(project='ee-weecologygsoc')
except Exception:
    ee.Authenticate(); ee.Initialize()

# reproject geometries to WGS84 for Earth Engine
g_o = gpd.GeoSeries(paired['geometry_orig'], crs=PROJ_CRS).to_crs(4326).values
g_c = gpd.GeoSeries(paired['geometry_corr'], crs=PROJ_CRS).to_crs(4326).values

def poly_to_mask(geom, bounds, W, H):
    """Rasterise a (Multi)Polygon to a boolean (H,W) pixel mask in tile space."""
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    for p in parts:
        xs, ys = p.exterior.xy
        px = [(x - bounds[0]) / (bounds[2] - bounds[0]) * W for x in xs]
        py = [(1 - (y - bounds[1]) / (bounds[3] - bounds[1])) * H for y in ys]
        d.polygon(list(zip(px, py)), fill=1)
    return np.array(im, dtype=bool)

def exg(img):
    R, G, B = [img[:, :, i].astype(np.float32) for i in range(3)]
    return 2 * G - R - B   # excess green; lower = browner

N = min(20, len(paired))   # check first 20 pairs (bump to len(paired) for all)
TILE_PX = 600
stats = []
for i in range(N):
    o, c = g_o[i], g_c[i]
    bounds = unary_union([o, c]).buffer(0.003).bounds
    roi = ee.Geometry.BBox(*bounds)
    naip = (ee.ImageCollection('USDA/NAIP/DOQQ')
            .filterBounds(roi).filterDate('2009-01-01', '2009-12-31')
            .mosaic().clip(roi))
    url = naip.getThumbURL({'min': 0, 'max': 255, 'dimensions': TILE_PX,
                            'bands': ['R', 'G', 'B'], 'format': 'png'})
    img = (mpimg.imread(io.BytesIO(urllib.request.urlopen(url).read()),
                        format='png') * 255).astype(np.uint8)[:, :, :3]
    H, W = img.shape[:2]

    gmask = poly_to_mask(c, bounds, W, H)
    rmask = poly_to_mask(o, bounds, W, H)
    e = exg(img)
    green_in = e[gmask].mean() if gmask.any() else np.nan
    red_only = e[rmask & ~gmask].mean() if (rmask & ~gmask).any() else np.nan
    background = e[~rmask & ~gmask].mean()
    stats.append(dict(id=paired.iloc[i]['DAMAGE_AREA_ID'],
                      exg_green=green_in, exg_red_only=red_only,
                      exg_background=background,
                      contrast=background - green_in))  # +ve = green is browner
    print(f"  pair {i:2d}: ExG inside-green={green_in:7.1f}  "
          f"red-only={red_only:7.1f}  background={background:7.1f}  "
          f"contrast={background - green_in:+6.1f}")

sdf = pd.DataFrame(stats)
print("\n--- VISIBILITY SUMMARY ---")
print(f"  median ExG inside green : {sdf.exg_green.median():.1f}")
print(f"  median ExG background   : {sdf.exg_background.median():.1f}")
print(f"  median contrast (bg - green): {sdf.contrast.median():+.1f}")
print("\n  If contrast is clearly positive (green is browner than background),")
print("  the damage IS visible -> segmentation is feasible. If ~0, the signal")
print("  isn't in these pixels -> flag to mentors (need higher-res imagery).")
sdf.to_csv('/content/drive/MyDrive/Data/day1_visibility_diagnosis.csv', index=False)
