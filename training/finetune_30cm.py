# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Fine-tune and validate on the self-labeled 30cm dataset
#
# U-Net (resnet34) on [R,G,B], initialised from the TreeFinder dead-tree encoder, scored with 5-fold
# spatially-blocked cross-validation. Epoch and decision threshold are both chosen on a nested inner
# slice of the training folds, so the scored fold is never looked at.
#
# Baselines printed every run: PRIOR-ECHO (score the raw ADS polygon against the label, no model) and
# ZERO-SHOT (init weights, no fine-tune).
#
# Known limitations: labels are partial, so IoU is a lower bound; tiles are RESIZED to 384 from
# windows of 180-1500 m, so effective resolution varies ~0.5-3.9 m/px across the dataset (see the
# effective-resolution diagnostic near the end); run-to-run sd (~0.02) exceeds most config effects,
# so report a mean over repeats, never one run.

# %%
# !pip install -q segmentation-models-pytorch "albumentations>=1.3,<2"

# %%
# Must be set before torch initialises CUDA or use_deterministic_algorithms cannot take effect.
import os
import shutil
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from datetime import datetime

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import time
# Logged into run_history: Colab hands out different GPU models between sessions, and different GPUs
# run different conv kernels, which is one candidate explanation for identical configs scoring
# differently. Without this column that hypothesis cannot be tested retrospectively.
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
print("=" * 66)
print(f"  DEVICE = {DEVICE}   (torch.cuda.is_available() = {torch.cuda.is_available()})")
print(f"  GPU    = {GPU_NAME}")
if DEVICE == "cpu":
    print("  No GPU detected. Training on CPU is very slow (a fold prints only when it finishes).")
    print("  On Colab, set Runtime -> Change runtime type -> GPU. On GPU expect ~1-3 min/fold.")
print("=" * 66)

# Four independent RNGs must be pinned: numpy (splits), torch (weights/shuffle), cuDNN (conv kernels),
# and Python's `random` — albumentations draws from that one, and seeding numpy/torch does not cover it.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    import warnings
    warnings.filterwarnings("once", message=".*does not have a deterministic implementation.*")
except Exception as _e:
    print("  note: torch.use_deterministic_algorithms unavailable on this torch build:", _e)
print("device:", DEVICE, "| albumentations", A.__version__, "| torch", torch.__version__)

# %% [markdown]
# ## Config

# %%
# USE_CROPS=True trains on build_crops_30cm.py output: every sample a 384px window at a CONSTANT
# 0.60 m/px, which is exactly the resolution unet_treefinder_best.pt was pretrained at. The whole-tile
# alternative resizes 180-1500 m windows to 384px, giving 0.61-3.68 m/px -- a 6.1x spread, median
# 2.4x coarser than the encoder expects. Measured: rho(m/px, IoU)=-0.305, p=0.003 on tiles that fired.
USE_CROPS = True             # the reported configuration; False reproduces the older whole-tile run
DRIVE = Path("/content/drive/MyDrive/Data")
# Crops live on Colab LOCAL disk (build_crops_30cm.py writes them there) — thousands of small files
# read far faster from /content than from Drive. Outputs and run_history stay on Drive so they
# survive a runtime restart and so every run lands in ONE comparable table.
DATA = Path("/content/seed30cm_crops") if USE_CROPS else DRIVE / "seed30cm"
HIST_DIR = DRIVE / "seed30cm"
IMG_DIR, PRIOR_DIR, MASK_DIR = DATA / "images", DATA / "priors", DATA / "masks"
NIR_DIR = DATA / "nir"
IGNORE_DIR = DATA / "ignore"
OUT = HIST_DIR / ("finetune30cm_outputs_crops" if USE_CROPS else "finetune30cm_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# The crops live on LOCAL disk, which a fresh Colab session starts empty. build_crops_30cm.py saves
# one zip to Drive for exactly this reason, so restore it here rather than failing 40 lines later
# with a confusing "no images found". Fail LOUDLY if the zip is missing: silently falling back to
# whole tiles would produce a plausible-looking run of the wrong configuration.
if USE_CROPS and not (DATA / "images").exists():
    _zip = DRIVE / "seed30cm_crops.zip"
    if not _zip.exists():
        raise SystemExit(f"USE_CROPS=True but neither {DATA}/images nor {_zip} exists.\n"
                         f"Run data_prep/build_crops_30cm.py first — it writes that zip.")
    print(f"unpacking {_zip} -> {DATA.parent} ...", flush=True)
    shutil.unpack_archive(str(_zip), str(DATA.parent))
    print(f"  restored {len(list((DATA/'images').glob('*.png')))} crop images", flush=True)
SSL_WEIGHTS = Path("/content/drive/MyDrive/Data/ssl_outputs/ssl_pretrained.pt")
# Init priority in build_model: SSL (4ch native) -> TreeFinder (60cm, ~3773 labels) -> ImageNet.
# An ABSENT file falls back silently to ImageNet, so check the "init from ..." line in the log.
INIT_WEIGHTS = Path("/content/drive/MyDrive/Data/TreeFinder/segmentation_outputs/unet_treefinder_best.pt")

SIZE = 384                   # whole-tile mode RESIZES to this; crop mode is already 384 (no resample)
N_FOLDS = 5

# ITERATION SPEED. None = score every fold (the only setting valid for a reported number). A list
# runs just those folds, so a config can be screened in a fraction of the time: folds are still built
# from all N_FOLDS, so each model still trains on ~80% of the data and the recipe is unchanged —
# only the test coverage shrinks. Do NOT reduce N_FOLDS itself to go faster; 2-fold trains each model
# on half the data, which is a different (worse) model, not a cheaper measurement of the same one.
# A partial run tags train_ver with "-folds<N>" so it can never pool with a full run in the history.
FOLDS_TO_RUN = None          # e.g. [0, 2] to screen on two geographically distinct folds
# An "epoch" is one pass over the training set, so the same epoch count is ~6x more gradient steps
# on the crop dataset (~1300 samples) than on the 206 whole tiles. Scale it down or the crop run
# takes hours AND trains far longer than the recipe every existing number was measured under.
FT_EPOCHS = 30 if USE_CROPS else 60
EVAL_EVERY = 3 if USE_CROPS else 5

BEST_EP_MIN_FRAC = 0.5       # epochs before this fraction of the budget cannot win (ep>=30 of 60)
BEST_EP_MARGIN   = 0.005     # a new epoch must beat the incumbent by this much to replace it

TTA       = True             # 8-way dihedral averaging at inference
TTA_INNER = False            # off for epoch ranking: runs ~12x per fold and only needs to rank

# The model outputs a probability per pixel; the threshold turns it into a yes/no decision. Tuning it
# per fold on inner-val is nested and adds no optimism, but the inner slice is small (19-28 tiles, of
# which ~29% are empty-GT and score NaN), so the argmax is noisy. The A/B block after cross-validation
# rescores the SAME predictions at a fixed 0.5, which is a paired test and costs no GPU time.
TUNE_THRESHOLD = True
THR_GRID = np.round(np.arange(0.20, 0.71, 0.05), 2)
AB_FIXED_THR = 0.5           # comparison threshold for the free A/B readout

# Repeat the WHOLE cross-validation with different seeds and average the out-of-fold probabilities.
# 1 = fast iteration (~10 min). 3 = the setting for any number you intend to publish (~30 min).
SEED_REPEATS = 1

# The stability readout only pools runs sharing this string, so different procedures are never mixed.
# 2026-07-22-cosine-bestepoch : cosine LR + best-epoch selection      (mean 0.176, sd 0.013, n=4)
# 2026-07-27-seeded-tta       : + random.seed fix, late-epoch guard, TTA, ignore-aware scoring
#                               (mean 0.191, sd 0.031, n=3)
# 2026-07-27-thr-tuned        : + per-fold threshold tuned on inner-val, per-run reseed
#                               (mean 0.134, sd 0.007, n=3)
# NOTE: scoring changed at seeded-tta (iou_dice gained the ignore mask). Proof in the log: the
# model-free prior_echo baseline stepped 0.0774 -> 0.0786 there. Do NOT compare IoU, precision or
# det_auc across that boundary — they are different statistics.
# 2026-08-05-crops060       : + fixed 0.60 m/px crops (USE_CROPS). Appended automatically so a crop
#                             run can never pool with a whole-tile run in the stability readout.
TRAIN_VER = ("2026-07-27-thr-tuned" + ("-crops060" if USE_CROPS else "")
             + ("" if FOLDS_TO_RUN is None else f"-folds{len(FOLDS_TO_RUN)}"))

GROUPED_CV = True            # spatially-blocked folds; random CV inflates scores via spatial leakage
RUN_ABLATION = False         # also train the with-prior arm (~3x runtime); settled negative
USE_NDVI = False             # tested at 142 and 206 tiles, did not help
PRIOR_DT_TAU = 16            # px at SIZE: falloff of the soft prior hint outside the polygon
# Ignored when manifest.csv is present — the manifest curates negatives itself. To train damage-only,
# drop the negative rows there instead.
INCLUDE_EMPTY_TILES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEAN = IMAGENET_MEAN + ((0.5,) if USE_NDVI else ())
STD = IMAGENET_STD + ((0.5,) if USE_NDVI else ())

# %% [markdown]
# ## Load labeled tiles

# %%
MANIFEST = DATA / "manifest.csv"
if MANIFEST.exists():
    man = pd.read_csv(MANIFEST); man["id"] = man["id"].astype(str).str.zfill(4)
    used = man[man["use"] == True]
    ids = [i for i in used["id"].tolist() if (IMG_DIR / f"{i}.png").exists()]
    print(f"manifest.csv: training on {len(ids)} curated tiles "
          f"(damage {int((used.role=='damage').sum())}, negative {int((used.role=='negative').sum())})")
else:
    ids = sorted(p.stem for p in MASK_DIR.glob("*.png"))
    print(f"no manifest.csv -> using all {len(ids)} masks")
if len(ids) < 5:
    print("  Note: very few labeled tiles — metrics will be noisy. Aim for ~30-50 tiles for stable numbers.")

def load(stem):
    """Read one tile's four aligned layers -> (rgb, gt, prior, weight).

    weight is 1 where a pixel counts and 0 inside annotator 'ignore' regions; it is applied to the
    loss AND to every metric. Masks resize with NEAREST so their 0/1 boundaries are not eroded.
    """
    rgb = np.array(Image.open(IMG_DIR / f"{stem}.png").convert("RGB").resize((SIZE, SIZE)), np.uint8)
    mp = MASK_DIR / f"{stem}.png"                 # a curated no-damage tile may have no saved mask
    gt = (np.array(Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 128) if mp.exists() \
        else np.zeros((SIZE, SIZE), bool)
    pp = PRIOR_DIR / f"{stem}.png"
    prior = (np.array(Image.open(pp).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 128) if pp.exists() \
        else np.zeros((SIZE, SIZE), bool)
    if USE_NDVI:
        npp = NIR_DIR / f"{stem}.png"
        if npp.exists():
            nir = np.array(Image.open(npp).convert("L").resize((SIZE, SIZE)), np.float32)
            R = rgb[..., 0].astype(np.float32)
            ndvi = (nir - R) / (nir + R + 1.0)                  # [-1,1], dead => low
            ndvi_u8 = np.clip((ndvi + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
        else:
            ndvi_u8 = np.zeros((SIZE, SIZE), np.uint8)
        rgb = np.dstack([rgb, ndvi_u8])
    ip = IGNORE_DIR / f"{stem}.png"
    ign = (np.array(Image.open(ip).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 128) \
        if ip.exists() else np.zeros((SIZE, SIZE), bool)
    weight = (~ign).astype(np.float32)
    weight[gt] = 1.0                                             # never ignore a labeled damage pixel
    return rgb, gt.astype(np.float32), prior.astype(np.float32), weight

X = []
for _k, _s in enumerate(ids):
    X.append(load(_s))
    if (_k + 1) % 25 == 0 or _k == len(ids) - 1:
        print(f"  loading tiles from Drive: {_k+1}/{len(ids)} ...", flush=True)
if not INCLUDE_EMPTY_TILES and MANIFEST.exists():
    print("Note: INCLUDE_EMPTY_TILES=False is ignored because manifest.csv is present — the manifest "
          "curates negatives, so all its tiles are kept. To train damage-only, remove the negative rows "
          "from manifest.csv instead.")
if not INCLUDE_EMPTY_TILES and not MANIFEST.exists():
    keep = [i for i, (_, g, _, _) in enumerate(X) if g.sum() > 0]
    dropped = len(X) - len(keep)
    ids = [ids[i] for i in keep]; X = [X[i] for i in keep]
    print(f"INCLUDE_EMPTY_TILES=False -> dropped {dropped} all-healthy tile(s); "
          f"training on {len(ids)} tiles that contain damage")
pos_frac = np.mean([g.mean() for _, g, _, _ in X]) if X else 0
print(f"avg damage coverage per tile: {pos_frac*100:.1f}%")
empty_idx = [i for i, (_, g, _, _) in enumerate(X) if g.sum() == 0]
if empty_idx:
    print(f"no-damage tiles kept as negatives: {len(empty_idx)}/{len(X)} "
          f"({100.0*len(empty_idx)/len(X):.0f}%)")


def valid_mask(i):
    """Pixels of tile i that count for scoring: everything except annotator-marked 'ignore' regions."""
    return X[i][3] > 0.5

_n_ign = sum(1 for i in range(len(X)) if not valid_mask(i).all())
_ign_frac = np.mean([1.0 - valid_mask(i).mean() for i in range(len(X))]) * 100 if X else 0.0
print(f"ignore regions: present on {_n_ign}/{len(X)} tiles, covering {_ign_frac:.2f}% of pixels on "
      f"average — these pixels are excluded from BOTH the loss and every metric.")
if _n_ign == 0:
    print("   (no ignore/ masks found, so ignore-aware scoring changes nothing on this dataset)")

try:
    _ix = pd.read_csv(DATA / "index.csv"); _ix["id"] = _ix["id"].astype(str).str.zfill(4)
    TILE_LATLON = {r["id"]: (float(r["lat"]), float(r["lon"])) for _, r in _ix.iterrows()}
    TILE_WINDOW_M = ({r["id"]: float(r["window_m"]) for _, r in _ix.iterrows()}
                     if "window_m" in _ix.columns else {})
    # Present only in the CROP dataset: which source tile each crop came from. Overlapping crops of
    # one tile are near-duplicates, so they must share a fold or they leak between train and test.
    TILE_SOURCE = ({r["id"]: str(r["source_id"]) for _, r in _ix.iterrows()}
                   if "source_id" in _ix.columns else {})
except Exception:
    TILE_LATLON = {}; TILE_WINDOW_M = {}; TILE_SOURCE = {}

def make_folds():
    """Build the folds ONCE so every arm is scored on the same split.

    GROUPED_CV clusters tiles by lat/lon, so a model is always tested on an area it never trained on.
    Damage comes in outbreaks, so under random folds near-duplicate neighbours land in both train and
    test and the score inflates (Roberts 2016; Ploton 2020). The blocked number is the honest one.
    """
    n = len(X); k = max(2, min(N_FOLDS, n))
    if GROUPED_CV:
        coords = np.array([TILE_LATLON.get(ids[i], (np.nan, np.nan)) for i in range(n)], float)
        have = ~np.isnan(coords).any(axis=1)
        if have.sum() >= k:
            try:
                from sklearn.cluster import KMeans
                lab = np.full(n, -1, int)
                # Cluster once per GROUP, then give every member its group's fold. A group is the
                # source tile when training on crops, and the tile itself otherwise -- so with no
                # source_id column this is exactly the previous per-tile clustering, same order,
                # same result. Order is first-appearance (not sorted) to keep KMeans init identical.
                grp = [TILE_SOURCE.get(ids[i], ids[i]) for i in range(n)]
                uniq = list(dict.fromkeys(g for g, h in zip(grp, have) if h))
                gpos = {}
                for i in range(n):
                    if have[i]:
                        gpos.setdefault(grp[i], coords[i])
                gxy = np.array([gpos[g] for g in uniq], float)
                kk = int(min(k, len(uniq)))
                glab = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit_predict(gxy)
                gmap = dict(zip(uniq, glab))
                for i in range(n):
                    if have[i]:
                        lab[i] = gmap[grp[i]]
                if TILE_SOURCE:
                    print(f"GROUPED_CV: clustering {len(uniq)} SOURCE TILES (not {n} crops) so "
                          f"overlapping crops of one tile cannot straddle train and test")
                miss = np.where(~have)[0]
                for j, i in enumerate(miss):
                    lab[i] = j % kk                    # round-robin the unlocated tiles
                if len(miss):
                    print(f"GROUPED_CV: {len(miss)} tile(s) have no lat/lon in index.csv — clustered "
                          f"the rest and spread these across folds (still no random-CV fallback)")
                fl = [np.where(lab == c)[0] for c in range(k) if (lab == c).any()]
                sizes = [len(f) for f in fl]
                print(f"GROUPED_CV: {len(fl)} spatially-blocked folds, sizes {sizes}")
                if max(sizes) > 3 * min(sizes):
                    print("   note: fold sizes are uneven because outbreaks cluster geographically. "
                          "The headline averages over TILES (not over folds), so this does not bias "
                          "the mean — it only makes the per-fold numbers noisier.")
                return fl
            except Exception as e:
                print("GROUPED_CV failed (needs scikit-learn + lat/lon); falling back to RANDOM "
                      "folds — the resulting score is optimistic, do not report it:", e)
        else:
            print("GROUPED_CV: too few tiles with lat/lon to cluster; falling back to RANDOM folds — "
                  "the resulting score is optimistic, do not report it")
    rng = np.random.default_rng(SEED)
    return np.array_split(rng.permutation(n), k)

FOLDS = make_folds()

# %% [markdown]
# ## Prior as a soft distance-transform hint (ablation arm only)

# %%
from scipy.ndimage import distance_transform_edt

def soft_prior_hint(prior_binary, tau=None):
    """Hard 0/1 ADS polygon -> smooth field in [0,1]: ~1 inside, decaying to 0 over ~tau px outside."""
    tau = PRIOR_DT_TAU if tau is None else tau
    b = np.asarray(prior_binary) > 0.5
    if not b.any():
        return np.zeros(b.shape, np.float32)
    sdf = distance_transform_edt(b) - distance_transform_edt(~b)          # +inside, -outside (px)
    return (0.5 * (np.tanh(sdf / float(tau)) + 1.0)).astype(np.float32)   # 0.5 on the boundary

def prior_channel(pr_tensor):
    """Augmented prior mask tensor (H,W) -> soft DT-hint input channel tensor (1,H,W)."""
    return torch.from_numpy(soft_prior_hint(pr_tensor.detach().cpu().numpy())).unsqueeze(0)

# %% [markdown]
# ## Augmentation + dataset

# %%
# `additional_targets` applies the SAME geometry to mask/prior/weight, otherwise they drift apart.
# Probabilities are written as explicit p= because in some albumentations versions the first
# positional argument is always_apply, not p.
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.5),
    A.Normalize(MEAN, STD),
    ToTensorV2(),
], additional_targets={"prior": "mask", "weight": "mask"})

val_tf = A.Compose([A.Normalize(MEAN, STD), ToTensorV2()],
                   additional_targets={"prior": "mask", "weight": "mask"})

class SeedDS(Dataset):
    """Serves items[idx] as (x, mask, weight). tf = train_tf when training, val_tf when scoring."""
    def __init__(self, items, idx, tf, use_prior):
        self.items, self.idx, self.tf, self.use_prior = items, idx, tf, use_prior
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        rgb, gt, prior, weight = self.items[self.idx[j]]
        o = self.tf(image=rgb, mask=gt, prior=prior, weight=weight)
        img, m, pr, w = o["image"], o["mask"], o["prior"], o["weight"]
        x = torch.cat([img, prior_channel(pr)], 0) if self.use_prior else img
        return x, m.unsqueeze(0).float(), w.unsqueeze(0).float()

# %% [markdown]
# ## Model, loss, metrics

# %%
def build_model(use_prior):
    in_ch = (4 if USE_NDVI else 3) + (1 if use_prior else 0)
    m = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=1)
    src = next((p for p in (SSL_WEIGHTS, INIT_WEIGHTS) if p.exists()), None)
    if src is not None:
        try:
            sd = torch.load(src, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            msd = m.state_dict()
            keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
            msd.update(keep)
            # Inflate the first conv when an extra channel changes the input count: copy pretrained RGB
            # filters into channels 0-2 and zero the rest, so the model starts equivalent to RGB-only.
            infl = 0
            for k, v in sd.items():
                if (k in msd and k not in keep and v.dim() == 4 and v.shape[1] == 3
                        and msd[k].shape[1] > 3 and v.shape[0] == msd[k].shape[0]
                        and v.shape[2:] == msd[k].shape[2:]):
                    w = torch.zeros_like(msd[k]); w[:, :3] = v; msd[k] = w; infl += 1
            m.load_state_dict(msd)
            print(f"  init from {src.name}: {len(keep)}/{len(msd)} tensors matched (in_ch={in_ch})"
                  + (f" + {infl} first-conv inflated (RGB kept, extra ch zero-init)" if infl else ""))
        except Exception as e:
            print(f"  weight load failed ({src.name}); using ImageNet init:", e)
    return m.to(DEVICE)

def dice_focal(logits, target, weight=None, alpha=0.25, gamma=2.0, smooth=1.0):
    """Focal + Dice. Focal concentrates gradient on hard pixels (damage is ~4% of the tile); Dice
    optimises region overlap directly. `weight` zeroes ignore pixels in BOTH terms.

    Note: alpha here scales the whole focal term rather than being the class-dependent alpha_t of
    Lin et al. 2017. Changing that is untested and is deliberately not in the current results.
    """
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    focal_map = alpha * (1 - torch.exp(-bce)) ** gamma * bce
    if weight is None:
        focal = focal_map.mean()
        inter = (prob * target).sum((1, 2, 3))
        denom = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
    else:
        focal = (focal_map * weight).sum() / weight.sum().clamp(min=1.0)
        inter = (prob * target * weight).sum((1, 2, 3))
        denom = (prob * weight).sum((1, 2, 3)) + (target * weight).sum((1, 2, 3))
    dice = 1 - (2 * inter + smooth) / (denom + smooth)
    return focal + dice.mean()

def iou_dice(pred, gt, valid=None):
    """Region IoU and Dice for ONE tile, restricted to `valid` (non-ignore) pixels.

    Returns (nan, nan) on an empty label so no-damage tiles are excluded by np.nanmean — otherwise an
    empty prediction scores a free 1.0 and rewards a model that predicts nothing. Those tiles are
    scored separately by the commission metric.
    """
    if valid is not None:
        pred = pred & valid
        gt = gt & valid
    if gt.sum() == 0:
        return np.nan, np.nan
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    return inter / union, 2 * inter / (pred.sum() + gt.sum())


# run_cv fills these. OOF_THR is per TILE (for scoring); OOF_THR_FOLDS is per FOLD (for deployment).
# They are different quantities: a tile-weighted mean of OOF_THR is dominated by the largest fold,
# which is not what you want as a single threshold for new imagery.
OOF_THR = {}
OOF_THR_FOLDS = {}
RUN_DIVERGED = {}                 # {use_prior: [fold numbers whose training did not converge]}
OOF_SCORED = {}                   # {use_prior: bool[n] — was this tile in a fold that actually ran}
RUN_FOLD_IDX = {}                 # {use_prior: [0-based indices of the folds that ran]}


def scored_mask(use_prior=False):
    """Tiles a fold actually predicted. Everything is scored unless FOLDS_TO_RUN subsets them."""
    return OOF_SCORED.get(use_prior, np.ones(len(X), bool))

def thr_of(i, use_prior=False):
    """Decision threshold for tile i. Returns 0.5 until run_cv has filled OOF_THR."""
    a = OOF_THR.get(use_prior)
    return 0.5 if a is None else float(a[i])

def thr_fold_mean(use_prior=False):
    """Unweighted mean of the per-fold thresholds — the single threshold to apply to NEW imagery."""
    return float(np.mean(OOF_THR_FOLDS.get(use_prior, [0.5])))


def _dihedral(t, k, flip):
    """Apply view (k, flip): optional left-right mirror, then k quarter-turns."""
    if flip:
        t = torch.flip(t, [-1])
    return torch.rot90(t, k, (-2, -1))

def _dihedral_inv(t, k, flip):
    """Undo _dihedral so all 8 predictions line up before averaging."""
    t = torch.rot90(t, -k, (-2, -1))
    if flip:
        t = torch.flip(t, [-1])
    return t

@torch.no_grad()
def predict_prob(model, x, tta=None):
    """Predict one tile -> HxW probabilities. x is CxHxW, already normalised by val_tf."""
    use_tta = TTA if tta is None else tta
    x = x.unsqueeze(0).to(DEVICE)
    if not use_tta:
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()
    acc = None
    for k in range(4):
        for flip in (False, True):
            p = torch.sigmoid(model(_dihedral(x, k, flip)))
            p = _dihedral_inv(p, k, flip)
            acc = p if acc is None else acc + p
    return (acc / 8.0)[0, 0].cpu().numpy()

@torch.no_grad()
def zero_shot(use_prior):
    """Baseline: init weights with NO fine-tuning, same TTA and ignore handling as the real model.

    Threshold fixed at 0.5 — this baseline has no fold and no inner-val slice to tune one on.
    """
    model = build_model(use_prior).eval()
    ious = np.zeros(len(X))
    for i in range(len(X)):
        rgb, gt, prior, _ = X[i]
        o = val_tf(image=rgb, mask=gt, prior=prior)
        x = torch.cat([o["image"], prior_channel(o["prior"])], 0) if use_prior else o["image"]
        prob = predict_prob(model, x)
        ious[i] = iou_dice(prob > 0.5, X[i][1].astype(bool), valid_mask(i))[0]
    return ious

# %% [markdown]
# ## Cross-validated fine-tuning

# %%
def run_cv(use_prior, rep=0):
    """For each fold: train on the others, pick epoch + threshold on a nested inner slice, predict
    this fold. Every tile gets one prediction from a model that never saw it.

    `rep` shifts every seed so repeat calls are independent runs.
    Returns (oof_iou, oof_dice, oof_pred). Thresholds land in OOF_THR / OOF_THR_FOLDS.
    """
    _s = SEED + 1000 * rep
    random.seed(_s); np.random.seed(_s); torch.manual_seed(_s); torch.cuda.manual_seed_all(_s)

    folds = FOLDS
    _run = ([i for i in range(len(folds))] if FOLDS_TO_RUN is None
            else [i for i in FOLDS_TO_RUN if 0 <= i < len(folds)])
    # NaN, not 0, for tiles no fold scored — 0 would be averaged in as a genuine failure.
    oof_iou = np.full(len(X), np.nan); oof_dice = np.full(len(X), np.nan)
    oof_pred = np.zeros((len(X), SIZE, SIZE), np.float32)
    oof_scored = np.zeros(len(X), bool)
    oof_thr = np.full(len(X), 0.5, np.float32)
    ep_choices = []
    thr_choices = []
    diverged = []                                # folds whose training loss rose instead of settling
    for f, val_idx in enumerate(folds):
        if f not in _run:
            continue
        oof_scored[val_idx] = True
        train_all = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        # Nested inner split: the epoch and threshold are decisions fitted to data, so they are made
        # on a slice carved out of the TRAINING folds, never on the fold being scored.
        _r = np.random.default_rng([_s, f, int(use_prior)])
        _perm = _r.permutation(len(train_all))
        _ncut = max(4, int(round(0.15 * len(train_all))))
        inner_val, train_idx = train_all[_perm[:_ncut]], train_all[_perm[_ncut:]]
        model = build_model(use_prior)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
        dl = DataLoader(SeedDS(X, train_idx, train_tf, use_prior), batch_size=4,
                        shuffle=True, drop_last=False)
        arm = "w/ soft-prior" if use_prior else "no prior"
        t0 = time.time()
        print(f"  [{arm}] fold {f+1}/{len(folds)}: training {len(train_idx)} tiles "
              f"(+{len(inner_val)} inner-val) x {FT_EPOCHS} ep on {DEVICE} ...", flush=True)

        def _inner_iou():
            """Rank epochs by inner-val IoU at the best threshold on the grid, so the ranking measures
            separability rather than calibration. TTA off here: this runs ~12x per fold."""
            model.eval(); _ds = SeedDS(X, inner_val, val_tf, use_prior)
            ps, gs, vms = [], [], []
            with torch.no_grad():
                for k, gi in enumerate(inner_val):
                    xx, _, _ = _ds[k]
                    ps.append(predict_prob(model, xx, tta=TTA_INNER))
                    gs.append(X[gi][1].astype(bool)); vms.append(valid_mask(gi))
            model.train()
            grid = THR_GRID if TUNE_THRESHOLD else [0.5]
            best = 0.0
            for t in grid:
                acc = [iou_dice(p > t, g, v)[0] for p, g, v in zip(ps, gs, vms)]
                acc = [a for a in acc if not np.isnan(a)]
                if acc:
                    best = max(best, float(np.mean(acc)))
            return best

        best_iou, best_state, best_ep = -1.0, None, -1
        _min_ep = int(BEST_EP_MIN_FRAC * FT_EPOCHS)
        _loss_min, _loss_last = float("inf"), float("nan")
        model.train()
        for ep in range(FT_EPOCHS):
            ep_loss = 0.0
            for x, m, w in dl:
                x, m, w = x.to(DEVICE), m.to(DEVICE), w.to(DEVICE)
                opt.zero_grad()
                loss = dice_focal(model(x), m, w)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            sched.step()
            _loss_last = ep_loss / max(1, len(dl))
            _loss_min = min(_loss_min, _loss_last)
            if (ep + 1) % EVAL_EVERY == 0 or ep == FT_EPOCHS - 1:
                iv = _inner_iou()
                if (ep + 1) >= _min_ep and iv > best_iou + BEST_EP_MARGIN:
                    best_iou, best_ep = iv, ep + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ep == 0 or (ep + 1) % max(1, FT_EPOCHS // 4) == 0:
                print(f"      ep {ep+1:>2}/{FT_EPOCHS}  loss={ep_loss/max(1,len(dl)):.3f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  inner-best={best_iou:.3f}@ep{best_ep}  "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"      -> restored best epoch {best_ep} of {FT_EPOCHS} "
                  f"(inner-val IoU {best_iou:.3f}; epochs before {_min_ep} not eligible)", flush=True)
        ep_choices.append(best_ep)
        # With cosine decay to zero the loss should settle at its minimum. If it ends materially
        # ABOVE the best value it reached, this fold did not converge, and its score describes a
        # failed optimisation rather than the method.
        if np.isfinite(_loss_last) and _loss_last > _loss_min * 1.02:
            diverged.append(f + 1)
            print(f"      !! fold {f+1} DID NOT CONVERGE: final loss {_loss_last:.3f} vs best "
                  f"{_loss_min:.3f} (+{100*(_loss_last/_loss_min-1):.1f}%)", flush=True)
        if not use_prior and rep == 0:
            torch.save(model.state_dict(), OUT / f"unet_30cm_fold{f+1}.pt")

        model.eval()
        # Threshold fitted on the same inner-val slice, with TTA ON — a threshold must be picked under
        # the inference procedure it will be applied with or it does not transfer.
        thr = 0.5
        if TUNE_THRESHOLD:
            ivs = SeedDS(X, inner_val, val_tf, use_prior)
            _p, _g, _v = [], [], []
            with torch.no_grad():
                for k, gi in enumerate(inner_val):
                    xx, _, _ = ivs[k]
                    _p.append(predict_prob(model, xx))
                    _g.append(X[gi][1].astype(bool)); _v.append(valid_mask(gi))
            _sc = []
            for t in THR_GRID:
                _vals = [iou_dice(p > t, g, v)[0] for p, g, v in zip(_p, _g, _v)]
                _vals = [x for x in _vals if not np.isnan(x)]
                _sc.append(np.mean(_vals) if _vals else np.nan)
            if np.any(np.isfinite(_sc)):
                thr = float(THR_GRID[int(np.nanargmax(_sc))])
            # How many inner tiles the argmax actually rested on, and how flat the grid was. A tiny
            # count with a tiny spread means the choice is noise, not evidence.
            _nu = int(np.sum([not np.isnan(iou_dice(p > 0.5, g, v)[0]) for p, g, v in zip(_p, _g, _v)]))
            _sp = float(np.nanmax(_sc) - np.nanmin(_sc)) if np.any(np.isfinite(_sc)) else np.nan
            print(f"      threshold {thr:.2f} chosen on {_nu} usable inner tile(s) "
                  f"(of {len(inner_val)}); grid spread {_sp:.3f}", flush=True)
        thr_choices.append(round(thr, 2))
        oof_thr[val_idx] = thr

        vds = SeedDS(X, val_idx, val_tf, use_prior)
        with torch.no_grad():
            for k, gi in enumerate(val_idx):
                x, _, _ = vds[k]
                prob = predict_prob(model, x)
                oof_pred[gi] = prob
                iou, dice = iou_dice(prob > thr, X[gi][1].astype(bool), valid_mask(gi))
                oof_iou[gi], oof_dice[gi] = iou, dice
        print(f"  [{arm}] fold {f+1}/{len(folds)} IoU={np.nanmean(oof_iou[val_idx]):.3f} "
              f"(threshold {thr:.2f}, {len(val_idx)} test tiles)")
    print(f"  [{'w/ soft-prior' if use_prior else 'no prior'}] epochs kept per fold: {ep_choices} "
          f"(all >= {int(BEST_EP_MIN_FRAC*FT_EPOCHS)} by construction)")
    # FT_EPOCHS is a budget, not a setting: the epoch that ships is chosen on inner-val, so extra
    # epochs only help if the choice is pressed against the ceiling. Say so explicitly rather than
    # leaving it to be eyeballed — this is the whole evidence for "should we train longer?".
    _at_ceiling = sum(1 for e in ep_choices if e >= FT_EPOCHS - 1)
    if _at_ceiling >= max(2, len(ep_choices) // 2):
        print(f"      !! {_at_ceiling}/{len(ep_choices)} folds chose epoch >= {FT_EPOCHS-1}: inner-val was "
              f"still improving when the budget ran out. FT_EPOCHS={FT_EPOCHS} is TRUNCATING — raise it.")
    else:
        print(f"      epoch budget OK: only {_at_ceiling}/{len(ep_choices)} fold(s) near the "
              f"{FT_EPOCHS}-epoch ceiling, so more epochs would not change the result.")
    print(f"  [{'w/ soft-prior' if use_prior else 'no prior'}] thresholds chosen per fold: {thr_choices}"
          + ("  (all 0.50 = tuning disabled)" if not TUNE_THRESHOLD else ""))
    # FOLD HEALTH. The headline is a TILE-weighted mean, so a large fold that trains badly moves it a
    # long way while a small one barely registers. Print both means and the per-fold weights, and
    # shout if any fold collapsed — otherwise a failed fold looks like "the method got worse".
    _fi = np.array([np.nanmean(oof_iou[folds[i]]) for i in _run], float)
    _fw = np.array([int((~np.isnan(oof_iou[folds[i]])).sum()) for i in _run], float)
    _med = float(np.nanmedian(_fi))
    _bad = [_run[j] + 1 for j, v in enumerate(_fi) if v < max(0.05, 0.4 * _med)]
    print(f"  [{'w/ soft-prior' if use_prior else 'no prior'}] per-fold IoU {np.round(_fi,3).tolist()} "
          f"on {_fw.astype(int).tolist()} scored tiles "
          f"(weights {np.round(_fw/_fw.sum(),2).tolist()})  folds {[i+1 for i in _run]}")
    print(f"      tile-weighted mean {np.nansum(_fi*_fw)/_fw.sum():.4f}  |  "
          f"unweighted fold mean {np.nanmean(_fi):.4f}")
    RUN_DIVERGED[use_prior] = list(diverged)
    if _bad or diverged:
        print(f"  !! FOLD HEALTH WARNING: collapsed fold(s) {_bad}; non-converged fold(s) {diverged}.")
        print( "     A single failed fold can dominate the headline. Do NOT compare this run against")
        print( "     others as if the configuration caused the difference — re-run it first.")
    OOF_THR[use_prior] = oof_thr
    OOF_THR_FOLDS[use_prior] = list(thr_choices)
    OOF_SCORED[use_prior] = oof_scored
    RUN_FOLD_IDX[use_prior] = list(_run)
    return oof_iou, oof_dice, oof_pred


def run_cv_repeated(use_prior):
    """Run the whole cross-validation SEED_REPEATS times and average the out-of-fold probabilities.

    Legitimate because each repeat is a complete cross-validation, so a tile is still only ever
    predicted by models that never saw it. This is NOT the 5-fold ensemble, which cannot be scored
    here at all (4 of its 5 models trained on every tile).
    """
    if SEED_REPEATS <= 1:
        return run_cv(use_prior, rep=0)
    if not TUNE_THRESHOLD:
        print("  !! SEED_REPEATS > 1 with TUNE_THRESHOLD = False. Averaging sharpens the predictions,\n"
              "     which makes a pinned 0.5 MORE sensitive to calibration shift, not less — this\n"
              "     combination measured worse run-to-run spread than a single run. Turn tuning on.")
    probs = np.zeros((len(X), SIZE, SIZE), np.float32)
    thrs = np.zeros(len(X), np.float64)
    fold_thrs = []
    per_rep = []
    for r in range(SEED_REPEATS):
        print(f"\n  --- repeat {r+1}/{SEED_REPEATS}  (seed {SEED + 1000*r}) ---", flush=True)
        _i, _d, _p = run_cv(use_prior, rep=r)
        probs += _p; thrs += OOF_THR[use_prior]
        fold_thrs.append(OOF_THR_FOLDS[use_prior])
        per_rep.append(float(np.nanmean(_i)))
        print(f"  repeat {r+1} region IoU = {per_rep[-1]:.3f}")
    probs /= SEED_REPEATS; thrs /= SEED_REPEATS
    OOF_THR[use_prior] = thrs
    OOF_THR_FOLDS[use_prior] = list(np.mean(np.array(fold_thrs, float), axis=0))
    iou = np.full(len(X), np.nan); dice = np.full(len(X), np.nan)
    _sc = scored_mask(use_prior)
    for i in range(len(X)):
        if _sc[i]:
            iou[i], dice[i] = iou_dice(probs[i] > thrs[i], X[i][1].astype(bool), valid_mask(i))
    print(f"\n  AVERAGED over {SEED_REPEATS} repeats: region IoU = {np.nanmean(iou):.3f}"
          f"   (individual repeats: {[round(v, 3) for v in per_rep]}, "
          f"sd {np.std(per_rep):.3f})")
    print("  The averaged number is the one to report — a single repeat is a draw from that spread.")
    return iou, dice, probs

HEAD = False                  # headline arm = no-prior; every report/figure/CSV reads results[HEAD]
zs_iou = zero_shot(HEAD)
n_nodmg = int(np.sum([g.sum() == 0 for _, g, _, _ in X]))
# Prior-echo: score the raw ADS prior against the label, no model. Beating it shows the model learned
# from imagery rather than echoing its input. Stricter than the gain over zero-shot.
prior_iou = np.array([iou_dice(X[i][2].astype(bool), X[i][1].astype(bool), valid_mask(i))[0]
                      for i in range(len(X))])
print(f"\nno-damage tiles (empty GT, excluded from IoU mean): {n_nodmg}/{len(X)}")
print(f"PRIOR-ECHO baseline (no model, prior vs label): region IoU mean={np.nanmean(prior_iou):.3f}")
print(f"ZERO-SHOT (init weights, no fine-tune):        region IoU mean={np.nanmean(zs_iou):.3f} "
      f"median={np.nanmedian(zs_iou):.3f}")

results = {}
for up in ([HEAD, True] if RUN_ABLATION else [HEAD]):
    print(f"\n=== Cross-validation: {'WITH soft-prior (DT)' if up else 'WITHOUT prior (HEADLINE)'} ===")
    results[up] = run_cv_repeated(up)
if not RUN_ABLATION:
    print("\n(with-prior ablation SKIPPED — settled negative, logged 5x in run_history.csv."
          " Set RUN_ABLATION=True for the final paper table.)")

# %% [markdown]
# ## Paired threshold A/B (zero GPU cost)
# Rescores the SAME out-of-fold probabilities under two decision rules. Because the predictions are
# identical, this isolates the threshold exactly — unlike comparing two training runs, where the
# run-to-run sd (~0.02) is larger than the effect being measured.

# %%
def _score_at(pm, thr_fn):
    """Rescore saved probability maps at a threshold. Returns (per-tile IoU, per-tile recall, n_silent)."""
    iou = np.full(len(X), np.nan); rec = np.full(len(X), np.nan); silent = 0
    _sc = scored_mask(HEAD)
    for i in range(len(X)):
        if not _sc[i]:
            continue                 # no fold predicted this tile; its all-zero map is not a result
        v = valid_mask(i)
        g = X[i][1].astype(bool) & v
        p = (pm[i] > thr_fn(i)) & v
        iou[i] = iou_dice(pm[i] > thr_fn(i), X[i][1].astype(bool), v)[0]
        if g.sum() > 0:
            rec[i] = (p & g).sum() / g.sum()
            silent += int(p.sum() == 0)
    return iou, rec, silent

_pm_ab = results[HEAD][2]
_ab = {}
for _nm, _fn in (("tuned per fold", lambda i: thr_of(i)),
                 (f"fixed {AB_FIXED_THR:.2f}", lambda i: AB_FIXED_THR)):
    _i, _r, _sl = _score_at(_pm_ab, _fn)
    _ab[_nm] = (float(np.nanmean(_i)), float(np.nanmean(_r)), _sl,
                [round(float(np.nanmean(_i[FOLDS[j]])), 3)
                 for j in RUN_FOLD_IDX.get(HEAD, range(len(FOLDS)))])

print("\n" + "=" * 80)
print("PAIRED THRESHOLD A/B — identical predictions, two decision rules")
print("=" * 80)
print(f"  {'rule':<18} {'IoU':>7} {'recall':>8} {'silent':>7}   per-fold IoU")
for _nm, (_iu, _rc, _sl, _pf) in _ab.items():
    print(f"  {_nm:<18} {_iu:7.4f} {_rc:8.4f} {_sl:7d}   {_pf}")
_delta = _ab[f"fixed {AB_FIXED_THR:.2f}"][0] - _ab["tuned per fold"][0]
iou_at_fixed = _ab[f"fixed {AB_FIXED_THR:.2f}"][0]
n_silent_tuned = _ab["tuned per fold"][2]
n_silent_fixed = _ab[f"fixed {AB_FIXED_THR:.2f}"][2]
print(f"\n  delta (fixed - tuned) = {_delta:+.4f}")
print( "    >= +0.03 : tuning is costing IoU. Check which fold supplies it in the per-fold row above.")
print( "    <  +0.01 : the threshold is NOT the cause of the drop vs earlier runs — look elsewhere.")
print( "    <= -0.01 : tuning is helping and the fixed-0.5 numbers in older runs were the optimistic ones.")
print("=" * 80)

for up, (iou, dice, _) in results.items():
    print(f"\n{'WITH soft-prior (DT)' if up else 'WITHOUT prior (HEADLINE)'}:  region IoU "
          f"mean={np.nanmean(iou):.3f} median={np.nanmedian(iou):.3f} | Dice mean={np.nanmean(dice):.3f}")
gain_zs = np.nanmean(results[HEAD][0]) - np.nanmean(zs_iou)
gain_echo = np.nanmean(results[HEAD][0]) - np.nanmean(prior_iou)
print(f"\n>>> HEADLINE (no-prior) region IoU={np.nanmean(results[HEAD][0]):.3f}  |  "
      f"vs zero-shot {gain_zs:+.3f}  |  vs prior-echo {gain_echo:+.3f}")
print("    ('vs prior-echo' tests whether the model added value beyond copying the prior.)")

# PER-SOURCE-TILE MEAN. In crop mode the headline is a mean over CROPS, and a big annotation yields
# far more crops than a small one (57 vs 1 in this dataset), so a handful of sites dominate it. The
# mean over SOURCE TILES gives every annotated site one vote, which is the claim a reader assumes.
# Report both; a large gap means the headline is being carried by a few large sites.
iou_by_source = np.nan
if TILE_SOURCE:
    _bysrc = {}
    for i in range(len(X)):
        if not np.isnan(results[HEAD][0][i]):
            _bysrc.setdefault(TILE_SOURCE.get(ids[i], ids[i]), []).append(results[HEAD][0][i])
    if _bysrc:
        _srcm = np.array([np.mean(v) for v in _bysrc.values()], float)
        _npc = np.array([len(v) for v in _bysrc.values()], float)
        iou_by_source = float(np.mean(_srcm))
        _pe_src = {}
        for i in range(len(X)):
            if not np.isnan(prior_iou[i]):
                _pe_src.setdefault(TILE_SOURCE.get(ids[i], ids[i]), []).append(prior_iou[i])
        _pe = float(np.mean([np.mean(v) for v in _pe_src.values()])) if _pe_src else np.nan
        print(f"\n    PER-SOURCE-TILE mean IoU = {iou_by_source:.3f} over {len(_srcm)} annotated sites "
              f"({_npc.min():.0f}-{_npc.max():.0f} crops each, median {np.median(_npc):.0f})")
        print(f"      vs per-crop {np.nanmean(results[HEAD][0]):.3f}. The per-site number gives every "
              f"site one vote; quote it when a site contributes many crops.")
        print(f"      per-site prior-echo {_pe:.3f} -> per-site gain {iou_by_source - _pe:+.3f}")

# Commission on the no-damage tiles: where the ADS prior fired but the imagery shows no damage, how
# much did the model still predict? This is the only metric with COMPLETE ground truth.
commis = None
if empty_idx:
    pe = results[HEAD][2]
    def _frac_fired(mask_bool, i):
        v = valid_mask(i)
        return float((mask_bool & v).sum()) / max(int(v.sum()), 1)
    commis = np.array([_frac_fired(pe[i] > thr_of(i), i) for i in empty_idx])
    prior_cov = np.array([_frac_fired(X[i][2].astype(bool), i) for i in empty_idx])
    print(f"\nNO-DAMAGE tiles (n={len(empty_idx)}): ADS prior fired on {prior_cov.mean()*100:.1f}% of "
          f"pixels on avg; model predicted damage on {commis.mean()*100:.2f}% "
          f"(worst tile {commis.max()*100:.1f}%).")
    print("    Lower = the model correctly stays quiet where the ADS polygon is wrong (rejects the prior).")

# Detection asks a different question from IoU: can the model tell a damage tile from a healthy one?
# Partial labels barely affect it, so it is far more stable than pixel IoU.
det_auc = det_ap = det_auc_free = np.nan
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _oof = results[HEAD][2]
    tile_score = np.array([float(((_oof[i] > thr_of(i)) & valid_mask(i)).sum()) / max(int(valid_mask(i).sum()), 1)
                           for i in range(len(X))])
    # Threshold-FREE version: mean predicted probability, never binarised. If det_auc drops while
    # det_auc_free holds, the threshold drifted; if both drop, the model genuinely got worse.
    tile_score_free = np.array([float((_oof[i] * valid_mask(i)).sum()) / max(int(valid_mask(i).sum()), 1)
                                for i in range(len(X))])
    tile_label = np.array([1 if X[i][1].sum() > 0 else 0 for i in range(len(X))])
    if tile_label.min() == 0 and tile_label.max() == 1:
        det_auc = roc_auc_score(tile_label, tile_score)
        det_ap = average_precision_score(tile_label, tile_score)
        det_auc_free = roc_auc_score(tile_label, tile_score_free)
        _base_rate = tile_label.mean()
        print(f"\nDETECTION (tile has damage y/n, {int(tile_label.sum())} damage vs "
              f"{int((1-tile_label).sum())} negative tiles):")
        print(f"    ROC-AUC = {det_auc:.3f}   PR-AUC = {det_ap:.3f}  (random baseline PR-AUC = {_base_rate:.3f})")
        print(f"    threshold-FREE ROC-AUC = {det_auc_free:.3f} (mean probability, never binarised).")
        print(f"    Compare with the pixel IoU ({np.nanmean(results[HEAD][0]):.3f}): a large gap means")
        print( "    strong detection but weak localization. det_auc is NOT comparable across train_ver.")
except Exception as _e:
    print("  (detection AUC skipped:", _e, ")")

# %% [markdown]
# ## Effective-resolution diagnostic
# Tiles are RESIZED to SIZE from windows of 180-1500 m, so metres-per-pixel varies ~8x across the
# dataset. A conv net has no scale invariance: a 5-8 m dead crown is 11-17 px at the fine end and
# 1.3-2 px at the coarse end. This block tests whether that predicts where the model fails. It needs
# no GPU and no retraining. If IoU collapses at the coarse end, the fix is cropping at native
# resolution rather than resizing, and this becomes a headline result rather than a caveat.

# %%
_mpp = np.array([TILE_WINDOW_M.get(ids[i], np.nan) / SIZE for i in range(len(X))], float)
_hd_iou = results[HEAD][0]
_hd_pred = results[HEAD][2]
_is_silent = np.array([bool(((_hd_pred[i] > thr_of(i)) & valid_mask(i)).sum() == 0) for i in range(len(X))])
_ok = np.isfinite(_mpp) & ~np.isnan(_hd_iou)
if _ok.sum() >= 10:
    print("\n" + "=" * 80)
    print("EFFECTIVE RESOLUTION vs PERFORMANCE  (damage tiles only)")
    print("=" * 80)
    print(f"  m/px range: {_mpp[_ok].min():.2f} - {_mpp[_ok].max():.2f} "
          f"(ratio {_mpp[_ok].max()/max(_mpp[_ok].min(), 1e-9):.1f}x)")
    _q = np.nanpercentile(_mpp[_ok], [0, 25, 50, 75, 100])
    print(f"  {'m/px bin':<18} {'n':>4} {'IoU':>7} {'recall_proxy':>13} {'silent':>7}")
    _rows_mpp = []
    for _b in range(4):
        _lo, _hi = _q[_b], _q[_b + 1]
        _sel = _ok & (_mpp >= _lo) & ((_mpp <= _hi) if _b == 3 else (_mpp < _hi))
        if _sel.sum() == 0:
            continue
        _si = float(_is_silent[_sel].mean() * 100)
        print(f"  {f'{_lo:.2f} - {_hi:.2f}':<18} {int(_sel.sum()):>4} {np.nanmean(_hd_iou[_sel]):7.3f} "
              f"{'':>13} {_si:6.1f}%")
        _rows_mpp.append((float(_lo), float(_hi), int(_sel.sum()), float(np.nanmean(_hd_iou[_sel])), _si))
    # Two DIFFERENT failures are being conflated if this is tested naively. A tile scores IoU 0 either
    # because the model outlined the wrong pixels, or because it predicted nothing at all. Those need
    # separate tests, and the pile of exact zeros also floods Spearman with ties and drags |rho| down.
    try:
        from scipy.stats import spearmanr, mannwhitneyu
        _nz = int((_hd_iou[_ok] == 0).sum())
        _rho, _pv = spearmanr(_mpp[_ok], _hd_iou[_ok])
        print(f"\n  (a) ALL damage tiles      rho(m/px, IoU) = {_rho:+.3f}  p = {_pv:.4f}   n={int(_ok.sum())}")
        print(f"      {_nz} of them score exactly 0 ({100*_nz/max(int(_ok.sum()),1):.0f}%), so this test is tie-dominated.")
        _fired = _ok & ~_is_silent
        if _fired.sum() >= 10:
            _rf, _pf = spearmanr(_mpp[_fired], _hd_iou[_fired])
            print(f"  (b) TILES THAT FIRED      rho(m/px, IoU) = {_rf:+.3f}  p = {_pf:.4f}   n={int(_fired.sum())}")
            print( "      = does coarser resolution blur the BOUNDARY, given the model found something?")
        _sm, _nm = _mpp[_ok & _is_silent], _mpp[_ok & ~_is_silent]
        if len(_sm) >= 5 and len(_nm) >= 5:
            _u, _pu = mannwhitneyu(_sm, _nm, alternative="two-sided")
            print(f"  (c) SILENT vs FIRED       median m/px {np.median(_sm):.2f} vs {np.median(_nm):.2f}  "
                  f"Mann-Whitney p = {_pu:.4f}")
            print( "      = does coarser resolution predict going silent ENTIRELY?")
        print( "  Read (b) and (c), not (a). If both are null, the resize is NOT the dominant")
        print( "  limitation and the bottleneck is data volume, not pixel scale.")
    except Exception as _e:
        print("  (resolution statistics skipped:", _e, ")")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(_mpp[_ok], _hd_iou[_ok], s=18, alpha=.7, color="#1f77b4")
    axes[0].set_xlabel("effective m/px (window_m / SIZE)"); axes[0].set_ylabel("region IoU")
    axes[0].set_title("Does coarser effective resolution hurt IoU?")
    _oks = np.isfinite(_mpp)
    axes[1].scatter(_mpp[_oks], _is_silent[_oks].astype(float) + np.random.default_rng(0).normal(0, .02, _oks.sum()),
                    s=18, alpha=.6, color="#d62728")
    axes[1].set_xlabel("effective m/px"); axes[1].set_ylabel("model predicted nothing (1 = silent)")
    axes[1].set_title("Where does the model go completely silent?")
    fig.tight_layout(); fig.savefig(OUT / "resolution_diagnostic.png", dpi=150); plt.show()
    print("  saved resolution_diagnostic.png")
    print("=" * 80)
else:
    _rows_mpp = []
    print("\n(effective-resolution diagnostic skipped — index.csv has no usable 'window_m' column)")

# %% [markdown]
# ## Figures + report

# %%
main = results[HEAD]
iou, dice, pred = main
order = np.argsort(-np.nan_to_num(iou, nan=-1.0)); show = list(order[:4]) + list(order[-2:])
fig, axes = plt.subplots(len(show), 5, figsize=(16, 3.1 * len(show)))
cols = ["30cm image", "ADS prior (orange)", "your label (green)", "prediction (blue)", "overlay"]
for r, i in enumerate(show):
    rgb, gt, prior, _ = X[i]; rgb = rgb[..., :3]; pr = pred[i] > thr_of(i)
    for c, ax in enumerate(axes[r]):
        ax.imshow(rgb)
        if c == 1: ax.contour(prior > .5, [.5], colors=["orange"], linewidths=1.2)
        elif c == 2:
            ov = np.zeros((SIZE, SIZE, 4)); ov[gt > .5] = [0, 1, 0, .4]; ax.imshow(ov)
        elif c == 3:
            ov = np.zeros((SIZE, SIZE, 4)); ov[pr] = [0, .4, 1, .5]; ax.imshow(ov)
        elif c == 4:
            ov = np.zeros((SIZE, SIZE, 4)); ov[pr] = [0, .4, 1, .5]; ax.imshow(ov)
            ax.contour(gt > .5, [.5], colors=["lime"], linewidths=1.2)
        if r == 0: ax.set_title(cols[c], fontsize=9, fontweight="bold")
        ax.axis("off")
    axes[r, 0].set_ylabel(f"IoU {iou[i]:.2f}" if not np.isnan(iou[i]) else "no-damage", fontsize=8)
fig.suptitle("Fine-tuned on self-labeled 30cm (lime=your label, blue=model)", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "finetune30cm_grid.png", dpi=150); plt.show()

if RUN_ABLATION:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([results[True][0], results[False][0]], bins=12,
            label=["with soft prior (DT)", "without prior"], color=["#2ca02c", "#999"])
    ax.set_xlabel("region IoU"); ax.legend(); ax.set_title("Does the DT soft-prior hint help?")
    fig.tight_layout(); fig.savefig(OUT / "finetune30cm_ablation.png", dpi=150); plt.show()

_rc_txt = "no-prior"
oof_pred_rc = results[HEAD][2]
recall = np.full(len(X), np.nan)
precision = np.full(len(X), np.nan)
area_ratio = np.full(len(X), np.nan)
for _i in range(len(X)):
    _v = valid_mask(_i)
    _gt = X[_i][1].astype(bool) & _v
    _pr = (oof_pred_rc[_i] > thr_of(_i)) & _v
    if _gt.sum() > 0:
        recall[_i] = (_pr & _gt).sum() / _gt.sum()
        # NaN, not 0.0, when the model predicted nothing — a silent tile is undefined precision, not
        # imprecise. NOTE this differs from the pre-2026-07-27 code, which recorded 0.0 and included
        # it, so `precision` is NOT comparable to rows logged before that. Hence the column rename.
        precision[_i] = ((_pr & _gt).sum() / _pr.sum()) if _pr.sum() else np.nan
        area_ratio[_i] = _pr.sum() / _gt.sum()
_degen = np.where((recall > 0.95) & (precision < 0.10))[0]
_clean = np.where(~((recall > 0.95) & (precision < 0.10)) & ~np.isnan(recall))[0]
print(f"\nRECALL on labeled damage ({_rc_txt}): mean={np.nanmean(recall):.3f} "
      f"median={np.nanmedian(recall):.3f}  (fair under partial labels)")
print(f"PRECISION on labeled damage (silent tiles EXCLUDED): mean={np.nanmean(precision):.3f} | "
      f"median predicted/label area ratio={np.nanmedian(area_ratio):.1f}x")
_n_scored = int((~np.isnan(recall)).sum())
_silent = int(np.sum(np.isnan(precision) & ~np.isnan(recall)))
if _silent and _n_scored:
    print(f"  like-for-like precision (silent tiles as 0, the pre-2026-07-27 definition): "
          f"{np.nanmean(precision) * (_n_scored - _silent) / _n_scored:.3f}")
_degen_ids = [ids[i] for i in _degen]
print(f"  paint-everything tiles (recall>0.95 AND precision<0.10): {len(_degen)}"
      f"/{_n_scored}" + (f" -> {_degen_ids}" if len(_degen) else ""))
if _silent:
    print(f"  silent tiles (model predicted nothing at all): {_silent} — counted as recall 0, "
          f"precision undefined (excluded from the precision mean)")
if len(_degen):
    print(f"  recall excluding those tiles: {np.nanmean(recall[_clean]):.3f} "
          f"(vs {np.nanmean(recall):.3f} including them) — the more conservative number to report.")

# --- Does label coverage predict performance? -------------------------------------------------
# A crop qualifies as "damage" at MIN_LABEL_PX=64, i.e. 0.04% of its pixels. The tempting move is to
# raise that floor and drop the thinly-labelled crops. That WOULD lift the reported IoU — but by
# deleting the hard cases, not by fixing them, which is score inflation and a reviewer will say so.
# So measure it instead of acting on it. The last column is the number you would report after
# deleting everything below each bin; if it climbs steeply, the "improvement" is pure selection.
_cov = np.array([float(X[i][1].astype(bool).mean()) for i in range(len(X))])
_cov_scored = ~np.isnan(_hd_iou)
if _cov_scored.sum():
    print("\n--- IoU vs LABEL COVERAGE (how much of the crop your annotation marks as damage) ---")
    print(f"  {'label % of crop':<16} {'n':>5} {'IoU':>7} {'recall':>8} {'silent':>8}   "
          f"{'IoU if bins below were DROPPED':>30}")
    _edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.01]
    for _lo, _hi in zip(_edges[:-1], _edges[1:]):
        _sel = _cov_scored & (_cov >= _lo) & (_cov < _hi)
        if not _sel.sum():
            continue
        _cum = _cov_scored & (_cov >= _lo)
        print(f"  {f'{_lo*100:.1f} - {_hi*100:.1f}':<16} {int(_sel.sum()):>5} "
              f"{np.nanmean(_hd_iou[_sel]):7.3f} {np.nanmean(recall[_sel]):8.3f} "
              f"{100*_is_silent[_sel].mean():7.1f}%   {np.nanmean(_hd_iou[_cum]):>30.3f}")
    print("  READ: a flat IoU column means the label floor is NOT what limits the score, and raising\n"
          "  it would only inflate the number. A rising one means thin labels are genuinely noisy\n"
          "  supervision — which is an argument for dropping them from TRAINING, never from scoring.")

L = [f"# 30cm fine-tune report — {datetime.now():%Y-%m-%d %H:%M}",
     f"- labeled tiles: {len(ids)} | size {SIZE} | {N_FOLDS}-fold CV | {FT_EPOCHS} epochs/fold",
     f"- avg damage coverage: {pos_frac*100:.1f}% | no-damage tiles excluded from IoU mean: {n_nodmg}/{len(ids)}",
     f"- PRIOR-ECHO baseline (no model, prior vs label): IoU mean {np.nanmean(prior_iou):.3f}",
     f"- ZERO-SHOT baseline (no fine-tune): IoU mean {np.nanmean(zs_iou):.3f} / median {np.nanmedian(zs_iou):.3f}"]
for up, (iu, di, _) in results.items():
    L.append(f"- {'WITH soft-prior (DT)' if up else 'WITHOUT prior'}: IoU mean {np.nanmean(iu):.3f} / "
             f"median {np.nanmedian(iu):.3f} | Dice mean {np.nanmean(di):.3f}")
L.append(f"- gain (headline no-prior): vs zero-shot {gain_zs:+.3f} | vs prior-echo {gain_echo:+.3f} "
         f"(prior-echo is the honest 'learned beyond copying the prior' number)")
L.append(f"- RECALL on labeled damage ({_rc_txt}): mean {np.nanmean(recall):.3f} / "
         f"median {np.nanmedian(recall):.3f} (fair metric under partial labels)")
L.append(f"- decision threshold: {'tuned per fold on inner-val' if TUNE_THRESHOLD else 'fixed 0.5'}, "
         f"per-fold {OOF_THR_FOLDS.get(HEAD, [])} (unweighted mean {thr_fold_mean(HEAD):.2f}) "
         f"| seed repeats averaged: {SEED_REPEATS}")
L.append(f"- PAIRED A/B: IoU {_ab['tuned per fold'][0]:.4f} tuned vs {iou_at_fixed:.4f} at fixed "
         f"{AB_FIXED_THR:.2f} (delta {_delta:+.4f}); silent tiles {n_silent_tuned} vs {n_silent_fixed}")
if _rows_mpp:
    L.append("- effective m/px vs IoU: " + "; ".join(f"{a:.2f}-{b:.2f} m/px n={c} IoU={d:.3f} silent={e:.0f}%"
                                                     for a, b, c, d, e in _rows_mpp))
if not np.isnan(det_auc_free):
    L.append(f"- DETECTION tile-level ROC-AUC {det_auc:.3f} (at the decision threshold) / "
             f"{det_auc_free:.3f} (threshold-free, mean probability)")
if commis is not None:
    L.append(f"- no-damage commission: model predicts damage on {commis.mean()*100:.2f}% of pixels on "
             f"{len(empty_idx)} empty tile(s) (lower=better; measures rejecting a spurious ADS prior)")
pd.DataFrame({"id": ids, "iou_priorEcho": prior_iou, "iou_zeroshot": zs_iou,
              "iou_no_prior": results[HEAD][0], "recall_no_prior": recall,
              "precision_nonsilent": precision, "area_ratio": area_ratio,
              "eff_m_per_px": _mpp, "silent": _is_silent.astype(int),
              "thr_used": [thr_of(i) for i in range(len(X))],
              **({"iou_soft_prior": results[True][0]} if RUN_ABLATION else {})}
             ).to_csv(OUT / "finetune30cm_metrics.csv", index=False)
(OUT / "run_report.md").write_text("\n".join(L))
print("\n".join(L)); print("saved figures + report to", OUT)

# %% [markdown]
# ## Self-descriptive readout

# %%
def _verdict(ok, ok_txt, warn_txt):
    return ok_txt if ok else warn_txt

_no_prior   = np.nanmean(results[HEAD][0])
_with_prior = np.nanmean(results[True][0]) if RUN_ABLATION else np.nan
_head       = _no_prior
_echo       = np.nanmean(prior_iou)
_beats_echo = _head - _echo
_commis     = (commis.mean() * 100) if commis is not None else np.nan
_recall     = np.nanmean(recall)
_folds_np   = results[HEAD][0]
_folds_show = np.round([np.nanmean(_folds_np[f]) for f in FOLDS], 3).tolist()

print("\n" + "=" * 80)
print("HOW TO READ THIS RUN  (plain-language targets)")
print("=" * 80)
print(f"1) Headline region IoU (no-prior) = {_head:.3f}")
print( "     Good  : 3-run mean rises past ~0.20 with sd < 0.02.")
print( "     Flat  : 3-run mean stays ~0.16 (data is no longer the limiting factor).")
print( "     Poor  : sd > 0.04 (training unstable — address first).")
print(f"2) vs prior-echo = {_beats_echo:+.3f}   (headline {_head:.3f} minus copying ADS {_echo:.3f})")
print("     " + _verdict(_beats_echo > 0.03,
      "Positive -> the model reads real damage from the imagery.",
      "Near or below 0 -> the model is mostly echoing the ADS polygon."))
print(f"3) Commission on {len(empty_idx) if empty_idx else 0} no-damage tiles = {_commis:.2f}%")
print("     " + (_verdict(_commis < 5,
      "Below 5% -> stays quiet where ADS is wrong.",
      "Above 15% -> over-predicts on healthy forest.") if not np.isnan(_commis) else "n/a"))
print(f"4) Recall on labeled damage = {_recall:.3f}   (target > 0.50; partial labels lower this)")
if RUN_ABLATION:
    print(f"5) Prior vs no-prior:  with={_with_prior:.3f}   no-prior={_no_prior:.3f}")
    print("     " + _verdict(_no_prior >= _with_prior - 0.005,
          "As expected: the prior as an input channel does not help.",
          "The prior helped this run; likely noise — rely on the multi-run mean."))
else:
    print("5) Prior ablation not run (off by default); the prior-echo baseline in (2) still runs.")
print(f"6) Fold spread (no-prior) = {_folds_show}")
print( "     Wide spread is expected under spatial CV. Near-identical folds can indicate leakage.")
_thr_used = np.asarray(OOF_THR.get(HEAD, [0.5]), dtype=float)
print(f"7) Thresholds: per-fold {OOF_THR_FOLDS.get(HEAD, [])}, unweighted mean {thr_fold_mean(HEAD):.2f}")
print(f"     (tile-weighted mean {_thr_used.mean():.2f} — dominated by the largest fold, which is why")
print( "      the deployed threshold uses the unweighted one.)")
print(f"8) Paired A/B delta (fixed {AB_FIXED_THR:.2f} minus tuned) = {_delta:+.4f}")
print( "     This is the ONLY threshold comparison here that is not swamped by run-to-run noise.")
print(f"9) Averaging repeats = SEED_REPEATS={SEED_REPEATS}"
      + ("  <- single run. Fine for iterating; set 3 for any number you publish."
         if SEED_REPEATS <= 1 else "  <- averaged; this is a publishable number."))
print("=" * 80)

# %% [markdown]
# ## Persistent run-history and learning curve

# %%
HIST = HIST_DIR / "run_history.csv"
row = {
    "timestamp": f"{datetime.now():%Y-%m-%d %H:%M}",
    "n_tiles": len(ids), "n_damage": int(len(ids) - n_nodmg), "n_negative": int(n_nodmg),
    "use_ndvi": USE_NDVI, "grouped_cv": GROUPED_CV, "include_empty": INCLUDE_EMPTY_TILES,
    "prior_dt_tau": PRIOR_DT_TAU, "tta": TTA,
    "seed_repeats": SEED_REPEATS,
    # Provenance for the reproducibility question: identical configs have scored 0.078-0.142, and a
    # changing GPU model (different conv kernels) is one candidate. A collapsed fold is another, and
    # a run with one must not be pooled with runs without one.
    "gpu": GPU_NAME,
    "diverged_folds": ",".join(str(v) for v in RUN_DIVERGED.get(HEAD, [])) or "none",
    # thr_mean stays TILE-weighted for continuity with rows already logged. thr_fold_mean is the
    # unweighted one used for deployment. Adding a column beats silently redefining an existing one.
    "thr_mean": round(float(np.mean(OOF_THR.get(HEAD, [0.5]))), 3),
    "thr_fold_mean": round(thr_fold_mean(HEAD), 3),
    "iou_no_prior":  round(float(_no_prior), 4),
    "iou_at_fixed_thr": round(float(iou_at_fixed), 4),     # paired A/B: same predictions at 0.5
    "ab_delta": round(float(_delta), 4),
    "n_silent": int(n_silent_tuned),
    "n_silent_fixed": int(n_silent_fixed),
    "iou_soft_prior": (round(float(_with_prior), 4) if RUN_ABLATION else np.nan),
    "dice":          round(float(np.nanmean(results[HEAD][1])), 4),
    "prior_echo":    round(float(_echo), 4),
    "iou_by_source": (round(float(iou_by_source), 4) if not np.isnan(iou_by_source) else np.nan),
    "folds_run":     (N_FOLDS if FOLDS_TO_RUN is None else len(FOLDS_TO_RUN)),
    "zero_shot":     round(float(np.nanmean(zs_iou)), 4),
    "recall":        round(float(_recall), 4),
    "commission_pct": (round(float(_commis), 2) if not np.isnan(_commis) else np.nan),
    "det_auc":       (round(float(det_auc), 4) if not np.isnan(det_auc) else np.nan),
    "det_auc_free":  (round(float(det_auc_free), 4) if not np.isnan(det_auc_free) else np.nan),
    "train_ver":     TRAIN_VER,
}
hist = (pd.concat([pd.read_csv(HIST), pd.DataFrame([row])], ignore_index=True)
        if HIST.exists() else pd.DataFrame([row]))
hist.to_csv(HIST, index=False)
print(f"\nappended this run to {HIST}   (total runs logged: {len(hist)})")
print(hist.to_string(index=False))

_yc = "iou_no_prior" if hist["iou_no_prior"].notna().any() else "iou_soft_prior"

# Pool only runs sharing this config AND train_ver, so different procedures are never mixed. A
# 3-repeat run is intrinsically steadier than a 1-repeat run, so those pools are kept separate too.
_tv = hist["train_ver"] if "train_ver" in hist.columns else pd.Series([np.nan] * len(hist))
_sr = (hist["seed_repeats"].fillna(1) if "seed_repeats" in hist.columns
       else pd.Series([1] * len(hist), index=hist.index))
_same = hist[(hist.n_damage == row["n_damage"]) & (hist.n_negative == row["n_negative"])
             & (hist.use_ndvi == row["use_ndvi"]) & (hist.grouped_cv == row["grouped_cv"])
             & (_sr == row["seed_repeats"]) & (_tv == TRAIN_VER)]
_sv = _same[_yc].dropna()
if len(_sv) >= 2:
    _lo, _hi, _sd = _sv.min(), _sv.max(), _sv.std()
    print(f"\nSTABILITY: {len(_sv)} runs share this config and train_ver='{TRAIN_VER}' -> {_yc} "
          f"mean={_sv.mean():.3f} sd={_sd:.3f} range=[{_lo:.3f}, {_hi:.3f}]")
    print(f"  => a between-run config change is real only if it moves the metric by more than ~{2*_sd:.3f}.")
    print( "  This is a WEAK test at n=3. The paired A/B above is the strong one for threshold questions.")
    if _sd > 0.02:
        print("  sd > 0.02 — training still noisy for single-run comparisons; consider lowering the LR.")
    else:
        print("  sd <= 0.02 — training is stable; A/B differences above ~0.04 are meaningful.")
else:
    print(f"\nSTABILITY: only {len(_sv)} run(s) with this config and the current train_ver "
          f"('{TRAIN_VER}'). Repeat it 2-3 times before relying on the number (runs with other "
          f"train_ver values are excluded — a different training procedure is a different experiment).")
    _prev = hist[(hist.n_damage == row["n_damage"]) & (hist.n_negative == row["n_negative"])
                 & (hist.use_ndvi == row["use_ndvi"]) & (hist.grouped_cv == row["grouped_cv"])
                 & (_sr == row["seed_repeats"]) & (_tv.notna()) & (_tv != TRAIN_VER)]
    for _pv, _grp in _prev.groupby(_prev["train_ver"]):
        _pvals = _grp[_yc].dropna()
        if len(_pvals) >= 2:
            print(f"  for reference, previous procedure '{_pv}': {len(_pvals)} runs, "
                  f"mean={_pvals.mean():.3f} sd={_pvals.std():.3f} "
                  f"-> this run is {_no_prior - _pvals.mean():+.3f} vs that mean "
                  f"({'outside' if abs(_no_prior - _pvals.mean()) > 2*_pvals.std() else 'within'} "
                  f"2 sd, so {'likely real' if abs(_no_prior - _pvals.mean()) > 2*_pvals.std() else 'not yet distinguishable from noise'})")
            print( "     CAUTION: scoring changed at the seeded-tta boundary, so IoU before and after "
                   "it are different statistics. Prefer the paired A/B for threshold questions.")
# RUN-SPREAD PLOT. The old "learning curve" plotted IoU against #damage-tiles, but 16 of 17 runs sit
# at the same tile count, so it drew a vertical line and read as a plunging curve. What the data
# actually support is a spread-per-procedure plot: every run, grouped by train_ver, against the
# do-nothing PRIOR-ECHO baseline. That is the honest figure and it is the one worth presenting.
_ds = hist[(hist.n_damage == row["n_damage"]) & (hist.n_negative == row["n_negative"])].dropna(subset=[_yc]).copy()
if len(_ds) >= 3:
    _ds["_tv"] = _ds["train_ver"].fillna("(pre-versioning)")
    _tvs = list(dict.fromkeys(_ds["_tv"].tolist()))
    fig, ax = plt.subplots(figsize=(1.9 * len(_tvs) + 3.5, 4.6))
    for j, tv in enumerate(_tvs):
        v = _ds.loc[_ds["_tv"] == tv, _yc].values.astype(float)
        jit = np.linspace(-.10, .10, len(v)) if len(v) > 1 else np.array([0.0])
        ax.scatter(np.full(len(v), j) + jit, v, s=55, alpha=.85, zorder=3,
                   color="#d62728" if tv == TRAIN_VER else "#1f77b4")
        ax.hlines(v.mean(), j - .28, j + .28, colors="k", linewidth=2, zorder=4)
        ax.annotate(f"n={len(v)}\nmean {v.mean():.3f}\nsd {v.std(ddof=1) if len(v)>1 else 0:.3f}",
                    (j, ax.get_ylim()[0]), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 2), textcoords="offset points")
    ax.axhline(float(row["prior_echo"]), color="grey", linestyle="--", linewidth=1.4,
               label=f"PRIOR-ECHO baseline ({row['prior_echo']:.3f}) — just trust the ADS polygon")
    ax.set_xticks(range(len(_tvs)))
    ax.set_xticklabels([t.replace("2026-", "") for t in _tvs], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel(f"region IoU ({_yc})")
    ax.set_title("Every run, by code version — spread vs the do-nothing baseline")
    ax.legend(fontsize=8, loc="upper right"); ax.margins(x=.12)
    fig.tight_layout(); fig.savefig(OUT / "run_spread.png", dpi=150); plt.show()
    print("saved run_spread.png  (replaces the old learning_curve.png, which needed varying tile counts)")
else:
    print("(run-spread plot appears once >=3 runs exist at this tile count.)")
if hist["n_damage"].nunique() >= 3:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    d = hist.dropna(subset=[_yc]).groupby("n_damage")[_yc].mean().reset_index()
    ax.plot(d["n_damage"], d[_yc], "o-", color="#1f77b4")
    ax.set_xlabel("# damage tiles labelled"); ax.set_ylabel(f"mean region IoU ({_yc})")
    ax.set_title("Learning curve (mean per tile count)")
    fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", dpi=150); plt.show()
    print("saved learning_curve.png")
else:
    print("(learning curve needs >=3 distinct #damage-tile counts; you have "
          f"{hist['n_damage'].nunique()} — label a batch at a different size to get one.)")

# %% [markdown]
# ## Contact sheet — every damage tile

# %%
# One sheet of 1908 crops is ~50 megapixels of thumbnails and legible at no zoom level. Write small
# PAGES instead, and lead with the three selections that are actually worth looking at.
SAVE_ALL_TILES = True
SHEET_COLS, SHEET_ROWS = 6, 6          # 36 per page, readable at 100%
SHEET_MODE = "all"                     # "all" = every crop, ~53 pages; "extremes" = best/worst only

if SAVE_ALL_TILES:
    hd_iou, _, hd_pred = results[HEAD]
    dmg = sorted([i for i in range(len(X)) if not np.isnan(hd_iou[i])], key=lambda i: -hd_iou[i])
    per_page = SHEET_COLS * SHEET_ROWS

    def _sheet(sel, fname, title):
        if not len(sel):
            return
        nrow = int(np.ceil(len(sel) / SHEET_COLS))
        fig, axes = plt.subplots(nrow, SHEET_COLS, figsize=(2.6 * SHEET_COLS, 2.85 * nrow))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for k, i in enumerate(sel):
            ax = axes[k // SHEET_COLS, k % SHEET_COLS]
            pr = hd_pred[i] > thr_of(i)
            ax.imshow(X[i][0][..., :3])
            ov = np.zeros((SIZE, SIZE, 4)); ov[pr] = [0, .4, 1, .5]; ax.imshow(ov)
            ax.contour(X[i][1] > .5, [.5], colors=["lime"], linewidths=.9)
            # A confirmed-healthy crop has no IoU (nothing to intersect). Label it by how much of the
            # crop the model wrongly painted, which is the only number that means anything there.
            ax.set_title(f"{ids[i]}  IoU {hd_iou[i]:.2f}" if not np.isnan(hd_iou[i])
                         else f"{ids[i]}  healthy — {100*pr.mean():.1f}% painted", fontsize=8)
        fig.suptitle(title + "   (lime = your label, blue = model)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}  ({len(sel)} crops)")

    print(f"contact sheets -> {OUT}")
    _sheet(dmg[:per_page], "sheet_best.png", f"BEST {min(per_page,len(dmg))} of {len(dmg)} damage crops")
    _sheet(dmg[-per_page:][::-1], "sheet_worst.png",
           f"WORST {min(per_page,len(dmg))} of {len(dmg)} damage crops")

    # The paint-everything crops: the model found the damage but flooded far past it. These are the
    # commission failures, and they cluster on a handful of SOURCE tiles — sort by source so the
    # pattern is visible rather than scattered across pages.
    _flood = sorted(_degen, key=lambda i: (TILE_SOURCE.get(ids[i], ids[i]), -hd_iou[i]))[:per_page]
    _sheet(_flood, "sheet_paint_everything.png",
           f"PAINT-EVERYTHING failures ({len(_degen)} total, showing {len(_flood)}), grouped by source tile")

    # The confirmed-NEGATIVE crops have never been drawn, only summarised as a commission %. They are
    # where the false alarms live, and a false alarm on ground verified healthy is the one error with
    # no "the label was incomplete" excuse. Worst first = largest predicted area on a clean crop.
    _neg = [i for i in range(len(X)) if np.isnan(hd_iou[i])]
    if _neg:
        _neg_area = {i: float(((hd_pred[i] > thr_of(i)) & valid_mask(i)).mean()) for i in _neg}
        _neg_sorted = sorted(_neg, key=lambda i: -_neg_area[i])
        _n_fired = sum(1 for i in _neg if _neg_area[i] > 0)
        _sheet(_neg_sorted[:per_page], "sheet_negatives_worst.png",
               f"FALSE ALARMS on confirmed-healthy crops ({_n_fired} of {len(_neg)} fired), worst first")
        if SHEET_MODE == "all":
            for p in range(int(np.ceil(len(_neg) / per_page))):
                _sheet(_neg_sorted[p * per_page:(p + 1) * per_page], f"sheet_neg_page{p+1:02d}.png",
                       f"Confirmed-healthy crops, most->least predicted — page {p+1}")

    if SHEET_MODE == "all":
        for p in range(int(np.ceil(len(dmg) / per_page))):
            _sheet(dmg[p * per_page:(p + 1) * per_page], f"sheet_page{p+1:02d}.png",
                   f"Damage crops, best->worst — page {p+1} of {int(np.ceil(len(dmg)/per_page))}")
    else:
        print(f"  (SHEET_MODE='extremes': {int(np.ceil(len(dmg)/per_page))} full pages not written. "
              f"Set SHEET_MODE='all' for every crop.)")

# %% [markdown]
# ## Final model on ALL tiles — for APPLYING to new data
#
# Cross-validation gives honest metrics but no deployable model, so one is retrained here on all tiles
# with the IDENTICAL recipe (cosine LR to 0) — otherwise the shipped model is one no reported metric
# describes. The five fold models are also saved; averaging them is usually better, and costs nothing.
#
# The ensemble CANNOT be scored on the tiles in this notebook: 4 of its 5 models trained on every
# tile. It is strictly for NEW imagery. To reduce noise in the REPORTED number use SEED_REPEATS.

# %%
SAVE_FINAL_MODEL = FOLDS_TO_RUN is None   # a screening run needs no deployable model (~14 min saved)
if not SAVE_FINAL_MODEL:
    print("final model SKIPPED (FOLDS_TO_RUN is set — this is a screening run, not a reportable one).")
if SAVE_FINAL_MODEL:
    final = build_model(False)
    fopt = torch.optim.AdamW(final.parameters(), lr=1e-4, weight_decay=1e-4)
    fsched = torch.optim.lr_scheduler.CosineAnnealingLR(fopt, T_max=FT_EPOCHS)
    fdl = DataLoader(SeedDS(X, np.arange(len(X)), train_tf, False), batch_size=4, shuffle=True)
    final.train(); _t0 = time.time()
    for ep in range(FT_EPOCHS):
        _el = 0.0
        for x, m, w in fdl:
            x, m, w = x.to(DEVICE), m.to(DEVICE), w.to(DEVICE)
            fopt.zero_grad(); _l = dice_focal(final(x), m, w); _l.backward(); fopt.step(); _el += _l.item()
        fsched.step()
        if ep == 0 or (ep + 1) % max(1, FT_EPOCHS // 4) == 0:
            print(f"  final model: ep {ep+1}/{FT_EPOCHS} loss={_el/max(1,len(fdl)):.3f} "
                  f"lr={fsched.get_last_lr()[0]:.2e} ({time.time()-_t0:.0f}s)", flush=True)
    _fp = OUT / "unet_30cm_final.pt"
    torch.save(final.state_dict(), _fp)
    print(f"saved final no-prior model -> {_fp}  (point colab_apply_to_monica.py WEIGHTS here)")
    print(f"  trained with the SAME recipe as cross-validation (cosine LR to 0, seed {SEED}).")
    _folds_saved = sorted(OUT.glob("unet_30cm_fold*.pt"))
    # UNWEIGHTED mean of the per-fold thresholds. A tile-weighted mean would let the largest fold
    # (38% of tiles here) set the threshold for all new imagery, which is not what it estimates.
    _thr_deploy = round(thr_fold_mean(False), 3)
    import json
    (OUT / "ensemble_meta.json").write_text(json.dumps({
        "train_ver": TRAIN_VER,
        "single_model": "unet_30cm_final.pt",
        "fold_models": [p.name for p in _folds_saved],
        "threshold": _thr_deploy,
        "threshold_per_fold": OOF_THR_FOLDS.get(False, []),
        "size": SIZE, "use_ndvi": USE_NDVI, "in_channels": (4 if USE_NDVI else 3),
        "tta": TTA,
        "note": ("Average the fold models' sigmoid probabilities, then threshold at 'threshold'. "
                 "Valid on NEW imagery only — every fold model was trained on 4/5 of the tiles in "
                 "this notebook, so the ensemble cannot be scored on them."),
    }, indent=2))
    print(f"  decision threshold for new data: {_thr_deploy:.2f} "
          f"(UNWEIGHTED mean of per-fold {OOF_THR_FOLDS.get(False, [])}; 0.5 is not the default any more)")
    if _folds_saved:
        print(f"  also saved {len(_folds_saved)} cross-validated fold models: "
              f"{[p.name for p in _folds_saved]}")
        print("  -> ensemble_meta.json written. Use for NEW imagery only.")


@torch.no_grad()
def ensemble_predict(x, models=None, thr=None):
    """Predict ONE new tile with the 5-fold ensemble -> (probability_map, binary_mask).

    x is CxHxW normalised by val_tf. thr defaults to the unweighted mean of the per-fold thresholds.
    """
    if models is None:
        models = []
        for p in sorted(OUT.glob("unet_30cm_fold*.pt")):
            m = build_model(False)
            m.load_state_dict(torch.load(p, map_location=DEVICE))
            models.append(m.eval())
    if not models:
        raise FileNotFoundError(f"no unet_30cm_fold*.pt in {OUT} — run the cross-validation first")
    if thr is None:
        thr = thr_fold_mean(False)
    prob = np.mean([predict_prob(m, x) for m in models], axis=0)
    return prob, prob > thr

# %% [markdown]
# ## Optional inference-time vegetation mask (off by default)
# Documented negative result: false-positive and true-positive pixels have nearly identical
# brightness and greenness in RGB (dead damage is itself brown/bright/low-green), so no threshold
# separates them and the mask removes true damage too. Enable VEG_MASK to reproduce.

# %%
from scipy.ndimage import binary_opening as _bopen, label as _cclabel

VEG_MASK  = False
BRIGHT_HI = 0.60     # mean-RGB (0-1) above which a non-green pixel is treated as bare ground
EXG_LO    = 0.10     # excess-green (2G-R-B) below which a pixel counts as "not green"
CLEAN_OPEN_IT  = 1       # binary-opening iterations; 0 = off
CLEAN_MIN_BLOB = 0.001   # drop blobs smaller than this fraction of the tile; 0 = off
PER_TILE_LOG   = True

def veg_keep_mask(rgb_u8):
    """True where a prediction is kept: drops bright non-green bare ground, keeps canopy and dark
    dead crowns."""
    rgb = rgb_u8[..., :3].astype(np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * G - R - B
    bright = (R + G + B) / 3.0
    return ~((bright > BRIGHT_HI) & (exg < EXG_LO))

def clean_pred(pr):
    """Morphological cleanup: shave fat boundaries and remove tiny speckles."""
    if CLEAN_OPEN_IT:
        pr = _bopen(pr, iterations=CLEAN_OPEN_IT)
    if CLEAN_MIN_BLOB:
        lab, n = _cclabel(pr)
        if n:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            pr = np.isin(lab, np.where(sz >= CLEAN_MIN_BLOB * pr.size)[0])
    return pr

if VEG_MASK:
    _, _, oof_np = results[HEAD]

    fb, fe, tb, te = [], [], [], []
    for i in range(len(X)):
        pr = oof_np[i] > thr_of(i)
        if not pr.any():
            continue
        gt = X[i][1].astype(bool)
        rgb = X[i][0][..., :3].astype(np.float32) / 255.0
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        b = (R + G + B) / 3.0; e = 2 * G - R - B
        fpm, tpm = pr & ~gt, pr & gt
        if fpm.any(): fb.append(b[fpm]); fe.append(e[fpm])
        if tpm.any(): tb.append(b[tpm]); te.append(e[tpm])
    if fb and tb:
        fb, fe = np.concatenate(fb), np.concatenate(fe)
        tb, te = np.concatenate(tb), np.concatenate(te)
        print("\n--- THRESHOLD CALIBRATION on YOUR pixels (predicted-damage pixels) ---")
        print(f"  FALSE-pos (outside your label): brightness p50={np.median(fb):.2f} p75={np.percentile(fb,75):.2f}"
              f" | exg p50={np.median(fe):.2f}")
        print(f"  TRUE-pos  (inside your label) : brightness p50={np.median(tb):.2f} p90={np.percentile(tb,90):.2f}"
              f" | exg p50={np.median(te):.2f}")
        print(f"  SUGGESTED BRIGHT_HI ~ {np.percentile(tb,90):.2f} (TRUE-pos p90 — above this is mostly bare ground)")
        print(f"  SUGGESTED EXG_LO    ~ {np.percentile(te,25):.2f} (TRUE-pos p25 — below this is mostly not-canopy)")
        print(f"  currently using BRIGHT_HI={BRIGHT_HI}, EXG_LO={EXG_LO}")

    rows_pt, iou3, rec3 = [], {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}
    for i in range(len(X)):
        _v = valid_mask(i)
        gt = X[i][1].astype(bool) & _v
        if gt.sum() == 0:
            continue
        pr = (oof_np[i] > thr_of(i)) & _v
        prm = pr & veg_keep_mask(X[i][0])
        prc = clean_pred(prm)
        for j, p in enumerate((pr, prm, prc)):
            iou3[j].append(iou_dice(p, gt)[0]); rec3[j].append((p & gt).sum() / gt.sum())
        rows_pt.append((ids[i], iou3[0][-1], iou3[1][-1], iou3[2][-1],
                        rec3[0][-1], rec3[2][-1], int((pr & ~veg_keep_mask(X[i][0])).sum())))
    com = []
    for p_fn in (lambda i: oof_np[i] > thr_of(i),
                 lambda i: (oof_np[i] > thr_of(i)) & veg_keep_mask(X[i][0]),
                 lambda i: clean_pred((oof_np[i] > thr_of(i)) & veg_keep_mask(X[i][0]))):
        com.append(np.mean([p_fn(i).mean() for i in empty_idx]) * 100 if empty_idx else np.nan)

    print("\n=== VEGETATION MASK + BOUNDARY CLEANUP (no-prior arm) ===")
    print( "                       raw     veg-masked   +cleaned")
    print(f"  region IoU  :      {np.mean(iou3[0]):.3f}      {np.mean(iou3[1]):.3f}       "
          f"{np.mean(iou3[2]):.3f}   (want UP)")
    print(f"  recall      :      {np.mean(rec3[0]):.3f}      {np.mean(rec3[1]):.3f}       {np.mean(rec3[2]):.3f}   (should barely drop)")
    print(f"  commission %:      {com[0]:.2f}       {com[1]:.2f}        {com[2]:.2f}   (want DOWN)")
    print(f"  settings: BRIGHT_HI={BRIGHT_HI} EXG_LO={EXG_LO} | CLEAN_OPEN_IT={CLEAN_OPEN_IT} "
          f"CLEAN_MIN_BLOB={CLEAN_MIN_BLOB}")

    if PER_TILE_LOG:
        rows_pt.sort(key=lambda r: -r[6])
        print("\n  PER-TILE (sorted by pixels the veg-mask removed) — tune with this:")
        print(f"    {'id':>6} {'IoU_raw':>8} {'IoU_veg':>8} {'IoU_cln':>8} {'rec_raw':>8} {'rec_cln':>8} {'px_removed':>11}")
        for r in rows_pt:
            print(f"    {r[0]:>6} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f} {r[4]:8.3f} {r[5]:8.3f} {r[6]:11d}")

    _rm = sorted(range(len(X)), key=lambda i: -((oof_np[i] > thr_of(i)) & ~veg_keep_mask(X[i][0])).sum())[:6]
    fig, axes = plt.subplots(len(_rm), 4, figsize=(12, 3 * len(_rm))); axes = np.atleast_2d(axes)
    for r, i in enumerate(_rm):
        rgb = X[i][0][..., :3]; pr = oof_np[i] > thr_of(i)
        prm = pr & veg_keep_mask(X[i][0]); prc = clean_pred(prm)
        for c, (ttl, ov) in enumerate([("image", None), ("raw prediction", pr),
                                       ("veg-masked", prm), ("+cleaned", prc)]):
            axes[r, c].imshow(rgb)
            if ov is not None:
                o = np.zeros((SIZE, SIZE, 4)); o[ov] = [0, .4, 1, .5]; axes[r, c].imshow(o)
            if r == 0: axes[r, c].set_title(ttl, fontsize=9, fontweight="bold")
            axes[r, c].axis("off")
    fig.suptitle("Veg-mask strips bare-ground FPs; cleanup shaves fat edges (blue=model)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "veg_mask_before_after.png", dpi=110); plt.show()
    print("  saved veg_mask_before_after.png")
