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
# # Dead-Tree Semantic Segmentation on TreeFinder (NAIP 60cm) — v2
#
# Train a U-Net to paint which pixels are dead/stressed trees. v2 fixes several
# issues found in a careful review of the v1 3-epoch run:
#
# - **NoData is now ignored** (not scored as "healthy"). Tiles can be ~46% black
#   padding; counting that as background diluted the ~1%-dead signal and inflated
#   metrics. We carry a `valid` mask and exclude NoData from BOTH loss and metrics.
# - **Loss = per-image Dice + Focal** (Focal replaces plain BCE, which — with 99%
#   background — pushed the model to predict "nothing" = low recall).
# - **Optimizer fixed:** AdamW with a *small* LR on the pretrained encoder (1e-4)
#   and a *larger* LR on the fresh decoder (1e-3), no weight-decay on BatchNorm/bias,
#   short warmup, cosine decay to a small floor, and gradient clipping.
# - **Honest metrics:** per-image (macro) Dice/recall, **per-object detection
#   recall** (% of annotated dead-tree blobs we find — what mentors care about),
#   and false-positive rate on healthy tiles.
#
# Plain-English terms: *Focal loss* = down-weights easy background so rare dead
# pixels drive learning. *Warmup* = start LR tiny so the pretrained encoder isn't
# wrecked in step 1. *Per-object recall* = of the dead-tree blobs labelled, what
# fraction did we detect at all.

# %% [markdown]
# ## 0 — Install & mount

# %%
# !pip install -q segmentation-models-pytorch albumentations rasterio torchmetrics

# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path

import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from scipy import ndimage
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from google.colab import drive
try:
    drive.mount('/content/drive')
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# ## 1 — Paths & config

# %%
DRIVE_ROOT = Path("/content/drive/MyDrive/Data/TreeFinder/")
IMG_DIR = DRIVE_ROOT / "tiles224_v3" / "tiles224_v3"      # (overwritten to local in 2.4)
CSV_PATH = DRIVE_ROOT / "tile_info224_v3.csv"
OUT_DIR = DRIVE_ROOT / "segmentation_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25              # <-- set to 6 for a quick PROBE first; if the curve rises, set 25
WARMUP_EPOCHS = 2
ENC_LR = 1e-4           # small LR for the pretrained encoder
DEC_LR = 1e-3           # larger LR for the randomly-initialised decoder
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
OP_THR = 0.30           # operating threshold for the recall-first metrics
ENCODER = "resnet34"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Optional extra input channels (TreeFinder band 4 = NIR). Both False =
# original RGB behaviour, unchanged. NIR/NDVI help separate truly-dead (low NIR)
# from shadow (also dark in RGB, but NOT low NIR). Flip on to train a variant. ---
USE_NIR = False         # append NIR (band 4) as an input channel
USE_NDVI = False        # append NDVI = (NIR-R)/(NIR+R) as an input channel
_EXTRA = (1 if USE_NIR else 0) + (1 if USE_NDVI else 0)
IN_CH = 3 + _EXTRA
CH_MEAN = tuple(IMAGENET_MEAN) + (0.5,) * _EXTRA    # extra channels ~[0,1], centre 0.5
CH_STD = tuple(IMAGENET_STD) + (0.5,) * _EXTRA

# %% [markdown]
# ## 2 — Index the data (positives + some negatives)

# %%
df = pd.read_csv(CSV_PATH)
print("columns:", df.columns.tolist())
df_pos = df[df["LabelSize"] > 0].copy()
print(f"tiles WITH dead trees : {len(df_pos)}")
df_neg_all = df[df["LabelSize"] == 0]
n_neg = min(len(df_neg_all), len(df_pos) // 4)
df_neg = df_neg_all.sample(n_neg, random_state=SEED) if n_neg > 0 else df_neg_all.iloc[:0]
print(f"negative tiles added  : {len(df_neg)}")
df_all = pd.concat([df_pos, df_neg]).reset_index(drop=True)

# %% [markdown]
# ## 2.4 — Copy ONLY the tiles we use to local disk (fast epochs)

# %%
import shutil
from concurrent.futures import ThreadPoolExecutor

USE_LOCAL_COPY = True
if USE_LOCAL_COPY:
    src_dir = IMG_DIR
    local_dir = Path("/content/tiles_local")
    local_dir.mkdir(parents=True, exist_ok=True)

    def _cp(fn):
        dst = local_dir / fn
        if dst.exists():
            return 0
        src = src_dir / fn
        if not src.exists():
            return -1
        shutil.copy(src, dst)
        return 1

    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(_cp, df_all["FileName"].tolist()))
    IMG_DIR = local_dir
    print(f"copied {sum(r==1 for r in res)} new; {sum(r==-1 for r in res)} missing; "
          f"IMG_DIR = {IMG_DIR}")

# %% [markdown]
# ## 2.5 — Confirm the mask/NoData encoding (band5: 1=dead, 255=NoData)

# %%
diag = df_pos["FileName"].head(8).tolist()
print(f"{'file':<26}{'LabelSize':>10}{'band5==1':>10}{'band5==255':>12}")
for fn in diag:
    with rasterio.open(IMG_DIR / fn) as src:
        m5 = src.read(5)
    ls = int(df_pos.loc[df_pos.FileName == fn, "LabelSize"].iloc[0])
    print(f"{fn[:26]:<26}{ls:>10}{int((m5==1).sum()):>10}{int((m5==255).sum()):>12}")
print("If band5==1 matches LabelSize, dead=(band5==1) and NoData=(band5==255) are correct.")

# %% [markdown]
# ## 3 — Split BY RAW IMAGE (no leakage)

# %%
group_col = "ImageRawID" if "ImageRawID" in df_all.columns else None
if group_col:
    rng = np.random.default_rng(SEED)
    ids = df_all[group_col].unique(); rng.shuffle(ids)
    val_ids = set(ids[:max(1, int(0.15 * len(ids)))])
    is_val = df_all[group_col].isin(val_ids)
else:
    is_val = np.random.default_rng(SEED).random(len(df_all)) < 0.15
df_train = df_all[~is_val].reset_index(drop=True)
df_val = df_all[is_val].reset_index(drop=True)
print(f"train {len(df_train)}  val {len(df_val)}")

# %% [markdown]
# ## 4 — Dataset: returns (image, dead-mask, valid-mask)
#
# `valid` = 1 where the pixel is real data, 0 where NoData padding. It rides
# through the SAME flips/rotations as the image (via `additional_targets`) so it
# stays aligned, and we use it to ignore NoData in the loss and metrics.

# %%
class TreeFinderSeg(Dataset):
    def __init__(self, frame, img_dir, transform):
        self.frame = frame
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def _read(self, path):
        with rasterio.open(path) as src:
            rgb = np.dstack([src.read(1), src.read(2), src.read(3)]).astype(np.float32)
            nir = (src.read(4).astype(np.float32)
                   if (src.count >= 4 and (USE_NIR or USE_NDVI)) else None)
            band5 = src.read(5) if src.count >= 5 else np.zeros(rgb.shape[:2], np.float32)
        if rgb.max() > 255:
            rgb = rgb / rgb.max() * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # band5: 1 = dead tree, 255 = NoData padding, 0 = background.
        # Belt-and-suspenders NoData: label sentinel OR black RGB.
        nodata = (band5 == 255) | ((rgb[..., 0] == 0) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0))
        dead = ((band5 == 1) & ~nodata).astype(np.float32)
        valid = (~nodata).astype(np.float32)
        if _EXTRA == 0:
            return rgb, dead, valid
        # Build a multi-channel image in [0,255] scale (A.Normalize divides by 255).
        nir_c = np.clip(nir, 0, 255) if nir is not None else np.zeros(rgb.shape[:2], np.float32)
        chans = [rgb.astype(np.float32)]
        if USE_NIR:
            chans.append(nir_c[..., None])
        if USE_NDVI:
            R = rgb[..., 0].astype(np.float32)
            ndvi = (nir_c - R) / (nir_c + R + 1e-6)         # [-1, 1]
            chans.append(((ndvi + 1.0) / 2.0 * 255.0)[..., None])   # -> [0, 255]
        img = np.concatenate(chans, axis=2).astype(np.float32)
        return img, dead, valid

    @staticmethod
    def _to_ch(m):
        t = m if isinstance(m, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(m))
        return t.float().unsqueeze(0)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        rgb, dead, valid = self._read(self.img_dir / row["FileName"])
        out = self.transform(image=rgb, mask=dead, valid=valid)
        return out["image"], self._to_ch(out["mask"]), self._to_ch(out["valid"])


_aug = dict(additional_targets={"valid": "mask"})
_norm = A.Normalize(mean=CH_MEAN, std=CH_STD, max_pixel_value=255.0)
_geo = [A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
# Brightness jitter only makes sense on RGB; skip it when NIR/NDVI channels are on.
_train_ops = _geo + ([A.RandomBrightnessContrast(p=0.3)] if _EXTRA == 0 else []) + [_norm, ToTensorV2()]
train_tf = A.Compose(_train_ops, **_aug)
val_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), _norm, ToTensorV2()], **_aug)

train_dl = DataLoader(TreeFinderSeg(df_train, IMG_DIR, train_tf), batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2, drop_last=True)
val_dl = DataLoader(TreeFinderSeg(df_val, IMG_DIR, val_tf), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=2)

# %% [markdown]
# ## 5 — Visualise the data (image + red dead-tree mask). NoData shows as black.

# %%
def denorm(t):
    # show RGB only (first 3 channels) even if NIR/NDVI were added
    x = t[:3].cpu().numpy().transpose(1, 2, 0)
    return np.clip(x * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)

def show_samples(ds, n=6, save=None):
    idxs = np.random.default_rng(0).choice(len(ds), min(n, len(ds)), replace=False)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6))
    for ax, i in zip(np.array(axes).flatten(), idxs):
        img, m, v = ds[i]
        ax.imshow(denorm(img))
        ov = np.zeros((*m.shape[1:], 4)); ov[m[0].numpy() > 0.5] = [1, 0, 0, 0.45]
        ax.imshow(ov)
        ax.set_title(f"dead {int(m.sum())} | valid {int(v.sum())}", fontsize=8); ax.axis("off")
    fig.suptitle("TreeFinder samples (red = dead-tree GT)", fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()

show_samples(TreeFinderSeg(df_val, IMG_DIR, val_tf), save=OUT_DIR / "01_data_samples.png")

# true dead fraction over VALID pixels only (honest imbalance number)
tot_dead = int(df_pos["LabelSize"].sum())
tot_valid = int((df_pos[["LabelSize"]].shape[0]) * IMG_SIZE * IMG_SIZE
                - df_pos.get("NoDataSize", pd.Series([0]*len(df_pos))).sum())
print(f"Dead pixels ~{100*tot_dead/max(tot_valid,1):.2f}% of VALID pixels in positive tiles.")

# %% [markdown]
# ## 6 — Model

# %%
model = smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet",
                 in_channels=IN_CH, classes=1).to(DEVICE)

# %% [markdown]
# ## 7 — Loss: per-image Dice + Focal, ignoring NoData
#
# - **Focal** (alpha=0.25, gamma=2): down-weights the flood of easy background so
#   the rare dead pixels actually drive the gradient (fixes low recall).
# - **Per-image soft Dice**: overlap score computed per tile (not pooled over the
#   whole batch), so tiles with few dead pixels aren't drowned out.
# - Both are averaged over **valid** pixels only (NoData ignored).

# %%
def masked_loss(logits, target, valid, alpha=0.25, gamma=2.0, smooth=1.0):
    prob = torch.sigmoid(logits)
    v = valid
    # --- Focal (pixelwise, masked) ---
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    focal = (focal * v).sum() / v.sum().clamp(min=1.0)
    # --- per-image soft Dice (masked) ---
    p, t = prob * v, target * v
    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    dice = 1.0 - (2 * inter + smooth) / (denom + smooth)
    return focal + dice.mean()

# %% [markdown]
# ## 8 — Optimizer (differential LR, no-decay on BN/bias), scheduler, metrics

# %%
def build_param_groups(model, enc_lr, dec_lr, wd):
    groups = {"enc_decay": [], "enc_nodecay": [], "dec_decay": [], "dec_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        enc = name.startswith("encoder")
        # BatchNorm params & biases are 1-D -> exclude from weight decay
        nod = (p.ndim == 1)
        key = ("enc" if enc else "dec") + ("_nodecay" if nod else "_decay")
        groups[key].append(p)
    return [
        {"params": groups["enc_decay"], "lr": enc_lr, "weight_decay": wd},
        {"params": groups["enc_nodecay"], "lr": enc_lr, "weight_decay": 0.0},
        {"params": groups["dec_decay"], "lr": dec_lr, "weight_decay": wd},
        {"params": groups["dec_nodecay"], "lr": dec_lr, "weight_decay": 0.0},
    ]

optimizer = torch.optim.AdamW(build_param_groups(model, ENC_LR, DEC_LR, WEIGHT_DECAY))
scheduler = SequentialLR(
    optimizer,
    schedulers=[LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS),
                CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)],
    milestones=[WARMUP_EPOCHS],
)

@torch.no_grad()
def evaluate(loader, thr=OP_THR):
    """Honest metrics: micro pixel IoU/Dice, per-image macro Dice/recall,
    per-OBJECT detection recall, and false-positive rate on healthy tiles.
    All computed over VALID (non-NoData) pixels only."""
    model.eval()
    micro_tp = micro_fp = micro_fn = 0
    dices, recalls = [], []
    obj_found = obj_total = 0
    neg_fp_pixels = neg_valid_pixels = 0
    loss_sum = 0.0
    for img, m, v in loader:
        img, m, v = img.to(DEVICE), m.to(DEVICE), v.to(DEVICE)
        logits = model(img)
        loss_sum += masked_loss(logits, m, v).item() * img.size(0)
        prob = torch.sigmoid(logits)
        vb = v > 0.5
        pred = (prob > thr) & vb
        tgt = (m > 0.5) & vb
        # micro counts
        micro_tp += (pred & tgt).sum().item()
        micro_fp += (pred & ~tgt & vb).sum().item()
        micro_fn += (~pred & tgt & vb).sum().item()
        # per-image
        for b in range(img.size(0)):
            pj, tj = pred[b, 0].cpu().numpy(), tgt[b, 0].cpu().numpy()
            tp = np.logical_and(pj, tj).sum()
            fp = np.logical_and(pj, ~tj).sum()
            fn = np.logical_and(~pj, tj).sum()
            if tj.sum() > 0:                                  # positive tile
                dices.append(2 * tp / (2 * tp + fp + fn + 1e-9))
                recalls.append(tp / (tp + fn + 1e-9))
                lbl, n = ndimage.label(tj)                    # per-object recall
                for k in range(1, n + 1):
                    obj_total += 1
                    if pj[lbl == k].any():
                        obj_found += 1
            else:                                             # negative tile
                neg_fp_pixels += fp
                neg_valid_pixels += (v[b, 0].cpu().numpy() > 0.5).sum()
    return dict(
        loss=loss_sum / len(loader.dataset),
        micro_iou=micro_tp / (micro_tp + micro_fp + micro_fn + 1e-9),
        micro_dice=2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn + 1e-9),
        macro_dice=float(np.mean(dices)) if dices else 0.0,
        macro_recall=float(np.mean(recalls)) if recalls else 0.0,
        obj_recall=obj_found / obj_total if obj_total else 0.0,
        neg_fp_rate=neg_fp_pixels / neg_valid_pixels if neg_valid_pixels else 0.0,
    )

# %% [markdown]
# ## 9 — Train
#
# TIP: set `EPOCHS = 6` first as a quick probe. If val macro-Dice rises and
# train loss falls, set `EPOCHS = 25` and re-run. We select the best checkpoint
# on **macro Dice** (aligned with what we report), not IoU@0.5.

# %%
history = {"train_loss": [], "val_loss": [], "macro_dice": [], "obj_recall": []}
best_metric = -1.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for img, m, v in train_dl:
        img, m, v = img.to(DEVICE), m.to(DEVICE), v.to(DEVICE)
        optimizer.zero_grad()
        loss = masked_loss(model(img), m, v)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        running += loss.item() * img.size(0)
    scheduler.step()
    tr = running / len(train_dl.dataset)
    ev = evaluate(val_dl)
    history["train_loss"].append(tr); history["val_loss"].append(ev["loss"])
    history["macro_dice"].append(ev["macro_dice"]); history["obj_recall"].append(ev["obj_recall"])
    print(f"ep {epoch:>2}/{EPOCHS} tr={tr:.3f} val={ev['loss']:.3f} "
          f"macroDice={ev['macro_dice']:.3f} objRecall={ev['obj_recall']:.3f} "
          f"microIoU={ev['micro_iou']:.3f} negFP={ev['neg_fp_rate']:.4f} "
          f"lr={optimizer.param_groups[0]['lr']:.2e}")
    if ev["macro_dice"] > best_metric:
        best_metric = ev["macro_dice"]
        torch.save(model.state_dict(), OUT_DIR / "unet_treefinder_best.pt")
print(f"\nBest val macro-Dice: {best_metric:.3f}  (saved)")

# %% [markdown]
# ## 10 — Curves

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, EPOCHS + 1)
ax[0].plot(ep, history["train_loss"], label="train"); ax[0].plot(ep, history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ep, history["macro_dice"], label="macro Dice")
ax[1].plot(ep, history["obj_recall"], label="object recall")
ax[1].set_title("Validation (higher=better)"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT_DIR / "02_training_curves.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 11 — Qualitative: image | GT | prediction | overlay

# %%
model.load_state_dict(torch.load(OUT_DIR / "unet_treefinder_best.pt")); model.eval()
val_ds = TreeFinderSeg(df_val, IMG_DIR, val_tf)

def show_predictions(ds, n=5, thr=OP_THR, save=None):
    idxs = np.random.default_rng(1).choice(len(ds), n, replace=False)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.1 * n))
    cols = ["NAIP", "Ground truth", "Prediction", "Overlay"]
    with torch.no_grad():
        for r, i in enumerate(idxs):
            img, m, v = ds[i]
            prob = torch.sigmoid(model(img.unsqueeze(0).to(DEVICE)))[0, 0].cpu().numpy()
            pred = (prob > thr) & (v[0].numpy() > 0.5)
            gt = m[0].numpy() > 0.5
            for c, ax in enumerate(axes[r]):
                if c == 0:
                    ax.imshow(denorm(img))
                elif c == 1:
                    ax.imshow(gt, cmap="Greens", vmin=0, vmax=1)
                elif c == 2:
                    ax.imshow(pred, cmap="Blues", vmin=0, vmax=1)
                else:
                    ax.imshow(denorm(img)); ov = np.zeros((*pred.shape, 4)); ov[pred] = [0, 0.4, 1, 0.5]
                    ax.imshow(ov); ax.contour(gt, levels=[0.5], colors=["lime"], linewidths=1.0)
                if r == 0:
                    ax.set_title(cols[c], fontweight="bold", fontsize=10)
                ax.axis("off")
    fig.suptitle("Dead-tree segmentation on held-out tiles (lime=GT, blue=pred)", fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()

show_predictions(val_ds, save=OUT_DIR / "03_predictions.png")

# %% [markdown]
# ## 12 — Threshold sweep (PR curve) + final honest summary

# %%
@torch.no_grad()
def sweep(loader, thrs):
    model.eval()
    counts = {t: [0, 0, 0] for t in thrs}
    for img, m, v in loader:
        img = img.to(DEVICE)
        prob = torch.sigmoid(model(img))[:, 0].cpu()
        vb = (m[:, 0] > 0.5), (v[:, 0] > 0.5)
        tgt, valid = vb
        for t in thrs:
            pred = (prob > t) & valid
            counts[t][0] += (pred & tgt).sum().item()
            counts[t][1] += (pred & ~tgt & valid).sum().item()
            counts[t][2] += (~pred & tgt & valid).sum().item()
    rows = []
    for t in thrs:
        tp, fp, fn = counts[t]
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        rows.append(dict(thr=t, precision=p, recall=r,
                         dice=2 * p * r / (p + r + 1e-9), iou=tp / (tp + fp + fn + 1e-9)))
    return pd.DataFrame(rows)

sw = sweep(val_dl, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
print(sw.to_string(index=False))
sw.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(sw.recall, sw.precision, "o-")
for _, r in sw.iterrows():
    ax[0].annotate(f"{r.thr:.2f}", (r.recall, r.precision), fontsize=7)
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title("PR curve"); ax[0].grid(alpha=0.3)
ax[1].plot(sw.thr, sw.dice, "s-", label="Dice"); ax[1].plot(sw.thr, sw.iou, "^-", label="IoU")
ax[1].set_xlabel("threshold"); ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_title("Overlap vs threshold")
fig.tight_layout(); fig.savefig(OUT_DIR / "04_pr_curve.png", dpi=150, bbox_inches="tight"); plt.show()

final = evaluate(val_dl, thr=OP_THR)
print("=" * 64)
print("  DEAD-TREE SEGMENTATION (TreeFinder) — HONEST SUMMARY")
print("=" * 64)
print(f"  train/val tiles      : {len(df_train)}/{len(df_val)}  (split by ImageRawID)")
print(f"  best val macro-Dice  : {best_metric:.3f}")
print(f"  per-OBJECT recall    : {final['obj_recall']:.3f}  (% of annotated dead blobs detected)")
print(f"  macro recall / dice  : {final['macro_recall']:.3f} / {final['macro_dice']:.3f}  @thr={OP_THR}")
print(f"  micro IoU            : {final['micro_iou']:.3f}")
print(f"  false-positive rate on HEALTHY tiles: {final['neg_fp_rate']:.4f}")
print("=" * 64)
print("  Note for mentors: TreeFinder GT is sparse, so pixel precision is an")
print("  UNDER-estimate (model may flag real-but-unlabelled dead trees). The")
print("  false-positive rate on healthy tiles is the honest over-prediction number.")

# %% [markdown]
# ## 13 — Save one copy-paste run report (share this with Claude / mentors)
#
# Writes `run_report.md` to the outputs folder: config + per-epoch table +
# threshold sweep + final summary, all in one place. Re-run this cell any time.

# %%
from datetime import datetime

def write_run_report(path=OUT_DIR / "run_report.md"):
    L = []
    L.append(f"# Segmentation run report — {datetime.now():%Y-%m-%d %H:%M}")
    L.append("\n## Config")
    L.append(f"- encoder: {ENCODER} (ImageNet) | epochs: {EPOCHS} | batch: {BATCH_SIZE}")
    L.append(f"- input channels: {IN_CH} (NIR={USE_NIR}, NDVI={USE_NDVI})")
    L.append(f"- enc_lr {ENC_LR} / dec_lr {DEC_LR} | wd {WEIGHT_DECAY} | "
             f"warmup {WARMUP_EPOCHS} | grad_clip {GRAD_CLIP}")
    L.append(f"- loss: per-image Dice + Focal (NoData ignored) | op_thr {OP_THR}")
    L.append(f"- train/val tiles: {len(df_train)}/{len(df_val)} (split by ImageRawID)")
    L.append("\n## Per-epoch")
    L.append("| epoch | train_loss | val_loss | macro_dice | obj_recall |")
    L.append("|---|---|---|---|---|")
    for i in range(len(history["train_loss"])):
        L.append(f"| {i+1} | {history['train_loss'][i]:.3f} | {history['val_loss'][i]:.3f} | "
                 f"{history['macro_dice'][i]:.3f} | {history['obj_recall'][i]:.3f} |")
    L.append("\n## Threshold sweep")
    L.append("| thr | precision | recall | dice | iou |")
    L.append("|---|---|---|---|---|")
    for _, r in sw.iterrows():
        L.append(f"| {r.thr:.2f} | {r.precision:.3f} | {r.recall:.3f} | "
                 f"{r.dice:.3f} | {r.iou:.3f} |")
    L.append("\n## Final summary")
    L.append(f"- best val macro-Dice: {best_metric:.3f}")
    L.append(f"- per-object recall: {final['obj_recall']:.3f}  (% of annotated dead blobs detected)")
    L.append(f"- macro recall / dice @thr={OP_THR}: {final['macro_recall']:.3f} / {final['macro_dice']:.3f}")
    L.append(f"- micro IoU: {final['micro_iou']:.3f}")
    L.append(f"- false-positive rate on healthy tiles: {final['neg_fp_rate']:.4f}")
    report = "\n".join(L)
    Path(path).write_text(report)
    print(report)
    print(f"\n>>> Saved copy-paste report to {path}")

write_run_report()
