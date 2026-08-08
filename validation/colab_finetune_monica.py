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
# # Fine-tune the TreeFinder model on Monica's 38 real pairs (cross-validated)
#
# Zero-shot transfer (TreeFinder 60cm -> Monica 1m) failed: the model predicted
# almost nothing (severe domain gap). This notebook tests the fix: **fine-tune**
# the TreeFinder-trained U-Net on the 38 real pairs so it adapts to the 1m 2009
# imagery AND to Monica's *region-style* labels.
#
# Because we only have 38 examples, we use **5-fold cross-validation**: train on
# ~30, test on the held-out ~8, rotate 5 times, so every pair gets an honest
# out-of-fold prediction. We compare **zero-shot vs fine-tuned region-IoU**.
#
# Includes a **domain-gap sanity check** (run the model on a TreeFinder tile via
# this exact code) to prove the zero-shot failure is a gap, not a bug.

# %%
# !pip install -q segmentation-models-pytorch albumentations rasterio geopandas earthengine-api

# %%
import io, urllib.request
import numpy as np, pandas as pd, geopandas as gpd
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from pathlib import Path
from shapely.ops import unary_union
from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure, label as cclabel
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from google.colab import drive, auth
try:
    drive.mount('/content/drive')
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEVICE)

# %% [markdown]
# ## 1 — Config

# %%
DRIVE = Path("/content/drive/MyDrive/Data")
WEIGHTS = DRIVE / "TreeFinder" / "segmentation_outputs" / "unet_treefinder_best.pt"
GDB = (DRIVE / "Historic_ADS_Data_Correct_GSC" / "Historic_ADS_Data_Correct_GSC" /
       "Data" / "OR_ADS_NAIP_Transform_2009.gdb")
TF_LOCAL = Path("/content/tiles_local")           # for the sanity check (if present)
OUT = DRIVE / "monica_finetune_outputs"; OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
WINDOW_M = 200.0            # ~0.9 m/px at 224px -> close to Monica's native 1m (avoid heavy upsampling)
N_FOLDS = 5
FT_EPOCHS = 40
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_model():
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    return m.to(DEVICE)

def load_tf_weights(m):
    m.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE)); return m

# %% [markdown]
# ## 2 — Load Monica's 38 pairs + Earth Engine

# %%
gdf_o = gpd.read_file(GDB, layer='OR_ADS_DAMAGE_AREA_R6_2009_Original')
gdf_c = gpd.read_file(GDB, layer='OR_ADS_DAMAGE_AREA_R6_2009_Transfrom_1')
orig = gdf_o[gdf_o['Checked'].isin(['Reshape', 'Resahpe'])]
corr = gdf_c[gdf_c['Transform'] == 'Reshape']
paired = orig[['DAMAGE_AREA_ID', 'geometry']].merge(
    corr[['DAMAGE_AREA_ID', 'geometry']], on='DAMAGE_AREA_ID', suffixes=('_o', '_c'))
PROJ = gdf_o.crs
print("pairs:", len(paired))

import ee
try:
    auth.authenticate_user(); ee.Initialize(project='ee-weecologygsoc')
except Exception:
    ee.Authenticate(); ee.Initialize()

def fetch_naip(bounds84, px=IMG_SIZE):
    roi = ee.Geometry.BBox(*bounds84)
    naip = (ee.ImageCollection('USDA/NAIP/DOQQ').filterBounds(roi)
            .filterDate('2009-01-01', '2009-12-31').mosaic().clip(roi))
    url = naip.getThumbURL({'min': 0, 'max': 255, 'dimensions': px,
                            'bands': ['R', 'G', 'B'], 'format': 'png'})
    return (mpimg.imread(io.BytesIO(urllib.request.urlopen(url).read()),
                         format='png')[:, :, :3] * 255).astype(np.uint8)

def sq_bounds(geom, win_m):
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = geom.centroid.x, geom.centroid.y
    half = max(win_m, (maxx - minx) * 1.3, (maxy - miny) * 1.3) / 2.0
    return cx - half, cy - half, cx + half, cy + half

def poly_mask(geom, b, W, H):
    im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    for p in ([geom] if geom.geom_type == 'Polygon' else list(geom.geoms)):
        xs, ys = p.exterior.xy
        px = [(x - b[0]) / (b[2] - b[0]) * W for x in xs]
        py = [(1 - (y - b[1]) / (b[3] - b[1])) * H for y in ys]
        d.polygon(list(zip(px, py)), fill=1)
    return np.array(im, bool)

# %% [markdown]
# ## 3 — Fetch & cache the 38 crops (image + Monica's green region mask)

# %%
CACHE = OUT / "monica_crops.npz"
if CACHE.exists():
    z = np.load(CACHE); X, Y = z["X"], z["Y"]
    print("loaded cached crops:", X.shape)
else:
    X, Y = [], []
    for i in range(len(paired)):
        gc = paired.iloc[i]['geometry_c']
        b = sq_bounds(gc, WINDOW_M)
        corners = gpd.GeoSeries(gpd.points_from_xy([b[0], b[2]], [b[1], b[3]]),
                                crs=PROJ).to_crs(4326)
        b84 = (corners.x.min(), corners.y.min(), corners.x.max(), corners.y.max())
        try:
            img = fetch_naip(b84)
        except Exception as e:
            print(f" pair {i} fetch failed: {e}"); img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        X.append(img); Y.append(poly_mask(gc, b84, IMG_SIZE, IMG_SIZE).astype(np.uint8))
        print(f" fetched {i+1}/{len(paired)}", end="\r")
    X, Y = np.stack(X), np.stack(Y)
    np.savez_compressed(CACHE, X=X, Y=Y)
    print("\ncached", X.shape)

print(f"green region covers ~{100*Y.mean():.1f}% of pixels on average "
      f"(much bigger than TreeFinder's ~1% -> easier to learn).")

# %% [markdown]
# ## 4 — Domain-gap sanity check: does the model fire on a TreeFinder tile?
#
# If the model produces clearly non-zero predictions on an in-domain TreeFinder
# tile through THIS code, the zero-shot failure on Monica is a genuine domain
# gap, not a preprocessing bug.

# %%
def normalize(img):
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

@torch.no_grad()
def predict_prob(model, img):
    return torch.sigmoid(model(normalize(img)))[0, 0].cpu().numpy()

sanity_model = load_tf_weights(build_model()).eval()
import rasterio
tf_tiles = list(TF_LOCAL.glob("*.tif"))[:3] if TF_LOCAL.exists() else []
if tf_tiles:
    print("Sanity check on TreeFinder tiles (in-domain):")
    for t in tf_tiles:
        with rasterio.open(t) as src:
            rgb = np.dstack([src.read(1), src.read(2), src.read(3)]).astype(np.uint8)
        p = predict_prob(sanity_model, rgb)
        print(f"  {t.name}: max prob={p.max():.3f}  mean={p.mean():.4f}")
    print("  -> if max prob is high (e.g. >0.5) in-domain but ~0 on Monica, "
          "the zero-shot failure is a DOMAIN GAP, not a bug.")
else:
    print("TreeFinder local tiles not found; skip sanity check "
          "(copy a few to /content/tiles_local to enable).")

# monica zero-shot probabilities for reference
zs_max = np.array([predict_prob(sanity_model, X[i]).max() for i in range(len(X))])
print(f"Monica zero-shot: median max-prob per tile = {np.median(zs_max):.3f} "
      f"(near 0 => model sees Monica as out-of-distribution)")

# %% [markdown]
# ## 5 — Loss, metrics, augmentation

# %%
def dice_focal(logits, target, alpha=0.25, gamma=2.0, smooth=1.0):
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    focal = (alpha * (1 - pt) ** gamma * bce).mean()
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + smooth) / (denom + smooth))
    return focal + dice.mean()

def region_iou(pred_bin, gt, k=4):
    st = generate_binary_structure(2, 2)
    reg = binary_closing(binary_dilation(pred_bin, st, iterations=k), st, iterations=k)
    inter = (reg & gt).sum(); union = (reg | gt).sum()
    return inter / union if union else 0.0

def plain_iou(pred_bin, gt):
    inter = (pred_bin & gt).sum(); union = (pred_bin | gt).sum()
    return inter / union if union else 0.0

train_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(0.5), A.VerticalFlip(0.5),
                      A.RandomRotate90(0.5), A.RandomBrightnessContrast(0.3),
                      A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])

class MonicaDS(torch.utils.data.Dataset):
    def __init__(self, X, Y, idx, tf):
        self.X, self.Y, self.idx, self.tf = X, Y, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        i = self.idx[j]
        o = self.tf(image=self.X[i], mask=self.Y[i].astype(np.float32))
        return o["image"], o["mask"].unsqueeze(0).float()

# %% [markdown]
# ## 6 — 5-fold cross-validated fine-tuning
#
# Each fold: start from TreeFinder weights, fine-tune on the training pairs,
# predict the held-out pairs (out-of-fold). We record region-IoU per pair.

# %%
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
folds = np.array_split(perm, N_FOLDS)

oof_iou_ft = np.zeros(len(X))
oof_iou_zs = np.zeros(len(X))
oof_pred = np.zeros((len(X), IMG_SIZE, IMG_SIZE), np.float32)

for f, val_idx in enumerate(folds):
    train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != f])
    model = load_tf_weights(build_model())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(MonicaDS(X, Y, train_idx, train_tf),
                                     batch_size=8, shuffle=True, drop_last=False)
    model.train()
    for ep in range(FT_EPOCHS):
        for img, m in dl:
            img, m = img.to(DEVICE), m.to(DEVICE)
            opt.zero_grad(); loss = dice_focal(model(img), m); loss.backward(); opt.step()
    # evaluate held-out fold
    model.eval()
    for i in val_idx:
        prob = predict_prob(model, X[i])
        oof_pred[i] = prob
        gt = Y[i].astype(bool)
        oof_iou_ft[i] = region_iou(prob > 0.5, gt)
        oof_iou_zs[i] = region_iou(predict_prob(sanity_model, X[i]) > 0.5, gt)
    print(f"fold {f+1}/{N_FOLDS}: fine-tuned region-IoU="
          f"{oof_iou_ft[val_idx].mean():.3f}  zero-shot={oof_iou_zs[val_idx].mean():.3f}")

print("\n" + "=" * 60)
print(f"  ZERO-SHOT   median region-IoU: {np.median(oof_iou_zs):.3f}  mean: {oof_iou_zs.mean():.3f}")
print(f"  FINE-TUNED  median region-IoU: {np.median(oof_iou_ft):.3f}  mean: {oof_iou_ft.mean():.3f}")
print(f"  improved on {(oof_iou_ft > oof_iou_zs).sum()}/{len(X)} pairs")
print("=" * 60)
pd.DataFrame({"id": paired['DAMAGE_AREA_ID'].values,
              "iou_zeroshot": oof_iou_zs, "iou_finetuned": oof_iou_ft}
             ).to_csv(OUT / "finetune_cv_metrics.csv", index=False)

# %% [markdown]
# ## 7 — Figure: image | Monica green | zero-shot | fine-tuned | overlay

# %%
order = np.argsort(-oof_iou_ft)
show = list(order[:4]) + list(order[-2:])
fig, axes = plt.subplots(len(show), 5, figsize=(16, 3.1 * len(show)))
cols = ["NAIP 2009", "Monica (green)", "Zero-shot", "Fine-tuned", "Overlay (fine-tuned)"]
for r, i in enumerate(show):
    gt = Y[i].astype(bool)
    zs = predict_prob(sanity_model, X[i]) > 0.5
    ft = oof_pred[i] > 0.5
    panels = [X[i], gt, zs, ft, X[i]]
    for c, ax in enumerate(axes[r]):
        if c == 0:
            ax.imshow(X[i])
        elif c == 1:
            ax.imshow(X[i]); ov = np.zeros((*gt.shape, 4)); ov[gt] = [0, 1, 0, .4]; ax.imshow(ov)
        elif c == 2:
            ax.imshow(zs, cmap="Blues", vmin=0, vmax=1)
        elif c == 3:
            ax.imshow(ft, cmap="Blues", vmin=0, vmax=1)
        else:
            ax.imshow(X[i]); ov = np.zeros((*ft.shape, 4)); ov[ft] = [0, .5, 1, .5]
            ax.imshow(ov); ax.contour(gt, [.5], colors=["lime"], linewidths=1.2)
        if r == 0:
            ax.set_title(cols[c], fontweight="bold", fontsize=9)
        ax.axis("off")
    axes[r, 0].set_ylabel(f"IoU {oof_iou_ft[i]:.2f}", fontsize=8)
fig.suptitle("Fine-tuning on 38 Monica pairs (5-fold CV) — lime = Monica, blue = model",
             fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "finetune_grid.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist([oof_iou_zs, oof_iou_ft], bins=12, label=["zero-shot", "fine-tuned"],
        color=["#999", "#2ca02c"])
ax.set_xlabel("region IoU vs Monica's green"); ax.legend()
ax.set_title("Zero-shot vs fine-tuned (out-of-fold)")
fig.tight_layout(); fig.savefig(OUT / "finetune_hist.png", dpi=150); plt.show()

# %% [markdown]
# ## 8 — Report
# If fine-tuned region-IoU >> zero-shot (~0), the pipeline works on real data
# once adapted: TreeFinder pre-training + fine-tuning on the 38 pairs. Next lever
# is Ben's self-supervised corrupt->recover pre-training to reduce reliance on
# the 38 labels.
