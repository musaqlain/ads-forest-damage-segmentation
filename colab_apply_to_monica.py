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
# # External validation — apply the trained model to Monica's 38 expert-corrected pairs
#
# This is an independent check. Every other metric is scored against the annotator's own labels; here
# the model is scored against a different expert (Monica), who manually re-aligned 38 ADS polygons on
# 2009 NAIP (1m). The model never saw her labels, so the agreement is not circular. Set `WEIGHTS` to
# the model to evaluate — the fine-tuned 30cm model (`unet_30cm_final.pt`, the default) or the
# zero-shot TreeFinder baseline — and every print/figure below labels itself from that file
# (`MODEL_NAME`) so results are not mislabeled.
#
# **The question:** does the model predict damage INSIDE Monica's corrected boundary more than outside?
# Two honest caveats we measure, not hide:
# - **Domain/granularity gap:** trained at 30cm on filled regions, tested at 1m — coarser, different
#   region. So raw pixel-IoU reads low even when the model is "right." We therefore lead with a
#   **density CONTRAST** (predicted-damage fraction inside minus outside her region) and report region
#   IoU as secondary.
# - **Silent pairs:** on some 1m tiles the 30cm model predicts nothing — we count those as "no
#   evidence", NOT agreement, and report the concentration rate only over pairs where it fired.

# %% [markdown]
# ## 0 — Install & mount

# %%
# !pip install -q segmentation-models-pytorch rasterio geopandas earthengine-api

# %%
import io, urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from shapely.ops import unary_union
from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp

from google.colab import drive, auth
try:
    drive.mount('/content/drive')
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# %% [markdown]
# ## 1 — Config: must match the training notebook

# %%
DRIVE = Path("/content/drive/MyDrive/Data")
# Two models can be compared on Monica's 38 expert pairs: the fine-tuned 30cm model vs the zero-shot
# TreeFinder baseline. Run the notebook once with each WEIGHTS and compare mean region_iou and contrast.
WEIGHTS = DRIVE / "seed30cm" / "finetune30cm_outputs" / "unet_30cm_final.pt"          # fine-tuned (no-prior, 3ch)
# WEIGHTS = DRIVE / "TreeFinder" / "segmentation_outputs" / "unet_treefinder_best.pt"  # zero-shot baseline
GDB = (DRIVE / "Historic_ADS_Data_Correct_GSC" / "Historic_ADS_Data_Correct_GSC" /
       "Data" / "OR_ADS_NAIP_Transform_2009.gdb")
OUT_DIR = DRIVE / "monica_apply_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 384              # match the 30cm fine-tune SIZE. With WINDOW_M=135 this is ~0.35 m/px, close to
                            # the 30cm training scale (U-Net is fully-conv, so any size works).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
# Window (metres) fetched per polygon. TreeFinder train tile = 224px * 0.6m = ~134m,
# so ~135m keeps the apparent scale similar to training. We enlarge if the polygon
# is bigger than the window.
WINDOW_M = 135.0
THR = 0.5                    # probability threshold for a "dead" pixel

# %% [markdown]
# ## 2 — Load the trained U-Net (same architecture as training)

# %%
model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=3, classes=1).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()
# MODEL_NAME follows the ACTUAL weights file, so the prints/titles/filenames below can never again
# say "zero-shot" while the fine-tuned model is loaded (they were hardcoded — a real mislabel risk).
MODEL_NAME = ("FINE-TUNED 30cm (unet_30cm_final)" if "final" in WEIGHTS.name
              else "ZERO-SHOT TreeFinder" if "treefinder" in WEIGHTS.name.lower()
              else WEIGHTS.stem)
MODEL_SLUG = "finetuned" if "final" in WEIGHTS.name else "zeroshot" if "treefinder" in WEIGHTS.name.lower() else WEIGHTS.stem
print("loaded weights from", WEIGHTS)
print(f"MODEL UNDER TEST: {MODEL_NAME}   <-- this is what every number/figure below describes")

# %% [markdown]
# ## 3 — Load Monica's 38 pairs (same as the diagnosis notebook)

# %%
gdf_o = gpd.read_file(GDB, layer='OR_ADS_DAMAGE_AREA_R6_2009_Original')
gdf_c = gpd.read_file(GDB, layer='OR_ADS_DAMAGE_AREA_R6_2009_Transfrom_1')
orig = gdf_o[gdf_o['Checked'].isin(['Reshape', 'Resahpe'])]
corr = gdf_c[gdf_c['Transform'] == 'Reshape']
paired = orig[['DAMAGE_AREA_ID', 'geometry', 'DCA_COMMON_NAME']].merge(
    corr[['DAMAGE_AREA_ID', 'geometry']], on='DAMAGE_AREA_ID',
    suffixes=('_orig', '_corr'))
PROJ = gdf_o.crs
# WGS84 copies for Earth Engine
g_o84 = gpd.GeoSeries(paired['geometry_orig'], crs=PROJ).to_crs(4326).values
g_c84 = gpd.GeoSeries(paired['geometry_corr'], crs=PROJ).to_crs(4326).values
print("pairs:", len(paired))

# %% [markdown]
# ## 4 — Earth Engine + helpers

# %%
import ee
try:
    auth.authenticate_user(); ee.Initialize(project='ee-weecologygsoc')
except Exception:
    ee.Authenticate(); ee.Initialize()

def fetch_naip(bounds, px=IMG_SIZE):
    """Fetch a 2009 NAIP RGB thumbnail for a WGS84 bbox -> (px,px,3) uint8."""
    roi = ee.Geometry.BBox(*bounds)
    naip = (ee.ImageCollection('USDA/NAIP/DOQQ').filterBounds(roi)
            .filterDate('2009-01-01', '2009-12-31').mosaic().clip(roi))
    url = naip.getThumbURL({'min': 0, 'max': 255, 'dimensions': px,
                            'bands': ['R', 'G', 'B'], 'format': 'png'})
    img = mpimg.imread(io.BytesIO(urllib.request.urlopen(url).read()), format='png')
    return (img[:, :, :3] * 255).astype(np.uint8)

def square_bounds_m(geom_proj, win_m):
    """Square window (in the projected metric CRS) centred on the polygon, at
    least win_m across (bigger if the polygon is larger)."""
    minx, miny, maxx, maxy = geom_proj.bounds
    cx, cy = geom_proj.centroid.x, geom_proj.centroid.y
    half = max(win_m, (maxx - minx) * 1.3, (maxy - miny) * 1.3) / 2.0
    return cx - half, cy - half, cx + half, cy + half

def poly_to_mask(geom, bounds, W, H):
    im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    for p in parts:
        xs, ys = p.exterior.xy
        px = [(x - bounds[0]) / (bounds[2] - bounds[0]) * W for x in xs]
        py = [(1 - (y - bounds[1]) / (bounds[3] - bounds[1])) * H for y in ys]
        d.polygon(list(zip(px, py)), fill=1)
    return np.array(im, bool)

@torch.no_grad()
def predict(img_rgb):
    """(H,W,3)uint8 -> per-pixel dead probability (H,W) float."""
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    return torch.sigmoid(model(t))[0, 0].cpu().numpy()

def region_iou(pred_bin, gt):
    """Merge sparse predicted blobs into a region, then IoU with the GT region."""
    st = generate_binary_structure(2, 2)
    reg = binary_closing(binary_dilation(pred_bin, st, iterations=4), st, iterations=4)
    inter = (reg & gt).sum(); union = (reg | gt).sum()
    return inter / union if union else 0.0

# %% [markdown]
# ## 5 — Run the model on all 38 pairs
#
# For each pair we crop a window centred on **Monica's corrected (green) polygon**
# (the feasibility test: "when looking at the right place, does the model find the
# damage she marked?"). We record, per pair:
# - `dens_in`  = mean predicted dead-probability INSIDE green
# - `dens_out` = mean predicted dead-probability OUTSIDE green
# - `contrast` = dens_in - dens_out  (>0 means the model concentrates on the damage)
# - `region_iou` = IoU after merging predicted blobs into a region

# %%
rows, cache = [], []
for i in range(len(paired)):
    gc_proj = paired.iloc[i]['geometry_corr']
    # window in projected CRS, then convert corners to WGS84 for EE
    bx = square_bounds_m(gc_proj, WINDOW_M)
    corners = gpd.GeoSeries(
        gpd.points_from_xy([bx[0], bx[2]], [bx[1], bx[3]]), crs=PROJ).to_crs(4326)
    bounds84 = (corners.x.min(), corners.y.min(), corners.x.max(), corners.y.max())
    try:
        img = fetch_naip(bounds84, IMG_SIZE)
    except Exception as e:
        print(f"  pair {i}: fetch failed ({e})"); continue

    green = poly_to_mask(g_c84[i], bounds84, IMG_SIZE, IMG_SIZE)
    prob = predict(img)
    pred_bin = prob > THR
    dens_in = float(prob[green].mean()) if green.any() else np.nan
    dens_out = float(prob[~green].mean())
    riou = region_iou(pred_bin, green)
    rows.append(dict(id=paired.iloc[i]['DAMAGE_AREA_ID'],
                     dca=paired.iloc[i]['DCA_COMMON_NAME'],
                     dens_in=dens_in, dens_out=dens_out,
                     contrast=dens_in - dens_out, region_iou=riou))
    cache.append((img, green, prob, pred_bin))
    print(f"  pair {i:2d}: dens_in={dens_in:.3f} dens_out={dens_out:.3f} "
          f"contrast={dens_in-dens_out:+.3f}  region_IoU={riou:.2f}")

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / f"monica_{MODEL_SLUG}_metrics.csv", index=False)

# %% [markdown]
# ## 6 — Aggregate result

# %%
_fired = res[res.dens_in + res.dens_out > 1e-6]        # pairs where the model predicted ANYTHING
_silent = len(res) - len(_fired)                        # all-zero pairs (model saw nothing — usually
                                                        #   the 1m-resolution gap, NOT agreement)
print("=" * 66)
print(f"  {MODEL_NAME}  on Monica's {len(res)} expert pairs")
print("=" * 66)
print(f"  median predicted-dead density INSIDE green : {res.dens_in.median():.3f}")
print(f"  median predicted-dead density OUTSIDE green: {res.dens_out.median():.3f}")
print(f"  median contrast (in - out)                 : {res.contrast.median():+.3f}")
print(f"  median region IoU (ALL pairs)              : {res.region_iou.median():.2f}")
print(f"  median region IoU (pairs it FIRED on)      : {_fired.region_iou.median():.2f}")
print(f"  model SILENT (predicted nothing)           : {_silent}/{len(res)} pairs "
      f"(likely the 1m-vs-30cm resolution gap — count as 'no evidence', NOT agreement)")
# HONEST concentration count: only among pairs where it actually fired, and require a REAL margin,
# not contrast > 0.000. The all-zero pairs have contrast==0 and must NOT be counted as 'concentrates'.
_conc = int((_fired.contrast > 0.02).sum())
print(f"  concentrates on the expert's damage        : {_conc}/{len(_fired)} pairs it fired on "
      f"(contrast>0.02); {(res.contrast>0).sum()}/{len(res)} if you count every hair above 0 "
      f"(that OVER-counts — includes the silent pairs).")
print("=" * 66)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(res.contrast.dropna(), bins=15, color="#1f77b4")
ax[0].axvline(0, color="red", ls="--"); ax[0].set_title("Prediction density: inside - outside green\n(>0 = model concentrates on the damage)")
ax[0].set_xlabel("contrast")
ax[1].hist(res.region_iou, bins=15, color="#2ca02c")
ax[1].set_title("Region IoU (pred region vs Monica's green)"); ax[1].set_xlabel("IoU")
fig.suptitle(f"{MODEL_NAME} vs Monica's expert corrections", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT_DIR / f"monica_{MODEL_SLUG}_summary.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7 — Per-pair figure: image | red (original) | green (Monica) | model | overlay

# %%
# SHOW_ALL=True renders EVERY pair (sorted best->worst by contrast) so nothing is cherry-picked — the
# honest version to share. False = a compact best-4 + worst-2 summary for slides. The figure gets very
# tall with all 38 pairs; it saves fine as a PNG you scroll or drop into a PDF.
SHOW_ALL = True
order = res.sort_values("contrast", ascending=False).index.tolist()
show = order if SHOW_ALL else order[:4] + order[-2:]
fig, axes = plt.subplots(len(show), 5, figsize=(16, 3.1 * len(show)))
cols = ["NAIP 2009", "Original (red)", "Monica corrected (green)", "Model prediction", "Overlay"]
for r, idx in enumerate(show):
    img, green, prob, pred_bin = cache[idx]
    bx = square_bounds_m(paired.iloc[idx]['geometry_corr'], WINDOW_M)
    corners = gpd.GeoSeries(gpd.points_from_xy([bx[0], bx[2]], [bx[1], bx[3]]),
                            crs=PROJ).to_crs(4326)
    b84 = (corners.x.min(), corners.y.min(), corners.x.max(), corners.y.max())
    red = poly_to_mask(g_o84[idx], b84, IMG_SIZE, IMG_SIZE)
    panels = [img, red, green, prob, img]
    for c, (ax, panel) in enumerate(zip(axes[r], panels)):
        if c in (1, 2):
            ax.imshow(img); ov = np.zeros((*panel.shape, 4))
            ov[panel] = [1, 0, 0, .4] if c == 1 else [0, 1, 0, .4]; ax.imshow(ov)
        elif c == 3:
            ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
        else:
            ax.imshow(img); ov = np.zeros((*pred_bin.shape, 4)); ov[pred_bin] = [0, 0.5, 1, .5]
            ax.imshow(ov); ax.contour(green, levels=[.5], colors=["lime"], linewidths=1.2)
        if r == 0:
            ax.set_title(cols[c], fontweight="bold", fontsize=9)
        ax.axis("off")
    axes[r, 0].set_ylabel(f"{res.loc[idx,'contrast']:+.2f}", fontsize=8)
fig.suptitle(f"{MODEL_NAME} on Monica's 2009 pairs "
             "(lime = Monica's boundary, blue = model)", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT_DIR / f"monica_{MODEL_SLUG}_grid.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8 — How to read this
#
# - **contrast > 0** (model predicts more damage inside Monica's corrected region than outside) is the
#   headline: an independent expert and the model agree on WHERE the damage is. Report it as
#   "concentrates in N/M pairs it fired on", not over all 38 (silent pairs are no-evidence).
# - **region IoU** reads modest (~0.2) because of the 1m-vs-30cm resolution gap and Monica drawing
#   filled regions vs the model's tighter blobs — expected, and why contrast is the primary metric.
# - **To compare two models** (fine-tuned vs zero-shot TreeFinder): re-run with each `WEIGHTS`; the
#   files auto-name by model, so `monica_finetuned_*` and `monica_zeroshot_*` sit side by side.
