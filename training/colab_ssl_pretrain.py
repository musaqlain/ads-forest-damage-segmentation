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
# # Self-supervised pre-training (Ben's corrupt→recover / CutMix idea)
#
# Goal: pre-train a segmenter on UNLABELED 30cm imagery so it needs fewer real
# labels. No human labels are used — we invent the answer:
#
# 1. **Weak label (free):** a vegetation-stress map from RGB (NGRDI + brightness
#    anomaly, reused from `coarse_align.py`) → threshold → a rough "damage" mask.
# 2. **Corrupt it:** randomly shift + rotate + dilate + blur that mask → a fake
#    "sloppy/misaligned annotation" = the **prior** channel.
# 3. **Train:** input `[R, G, B, corrupted_prior]` → recover the (clean) weak mask.
#
# The model learns *"given imagery + a roughly-placed annotation, snap it to the
# real signal"* — exactly the deployment task. Output weights (`ssl_pretrained.pt`)
# initialise `finetune_30cm.py`.
#
# NOTE: weak labels are noisy by design; SSL only needs them to teach the *shape*
# of the task. Real accuracy comes from the fine-tune step.

# %%
# !pip install -q segmentation-models-pytorch

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import (uniform_filter, gaussian_filter, shift as nd_shift,
                           binary_dilation, rotate as nd_rotate)
import segmentation_models_pytorch as smp

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEVICE)

# %% [markdown]
# ## Config

# %%
# Folder of UNLABELED 30cm RGB tiles (PNG). The seed tiles work; more is better.
TILES_DIR = Path("/content/drive/MyDrive/Data/seed30cm/images")
OUT = Path("/content/drive/MyDrive/Data/ssl_outputs"); OUT.mkdir(parents=True, exist_ok=True)

SIZE = 512
EPOCHS = 20
BATCH = 6
STRESS_PCTL = 88.0          # top (100-pctl)% most-stressed pixels become weak "damage"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# %% [markdown]
# ## Weak label (RGB vegetation stress) + corruption

# %%
def stress_map(rgb):
    """RGB (H,W,3) uint8 -> stress in [0,1] (reused from coarse_align.py logic)."""
    img = rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    ngrdi = (R - G) / (R + G + 1.0)
    stress = np.clip((ngrdi + 1.0) / 2.0 - 0.4, 0.0, 0.6) / 0.6   # browner => higher
    bright = (R + G + B) / 3.0
    lm = uniform_filter(bright, 31)
    lv = uniform_filter(bright ** 2, 31) - lm ** 2
    anomaly = np.clip(np.abs(bright - lm) / np.sqrt(np.maximum(lv, 1.0)) / 3.0, 0, 1)
    return (0.7 * stress + 0.3 * anomaly).astype(np.float32)


def weak_label(rgb):
    s = stress_map(rgb)
    thr = np.percentile(s, STRESS_PCTL)
    m = (s > thr).astype(np.uint8)
    return binary_dilation(m, iterations=1).astype(np.float32)


def corrupt(mask, rng):
    """Make a sloppy/misaligned 'prior' from a clean mask."""
    m = mask.copy()
    ang = rng.uniform(-15, 15)
    m = nd_rotate(m, ang, reshape=False, order=0)
    dy, dx = rng.uniform(-40, 40, size=2)          # shift in px (~12m at 30cm)
    m = nd_shift(m, (dy, dx), order=0)
    if rng.random() < 0.7:                          # usually make it looser/bigger
        m = binary_dilation(m, iterations=int(rng.integers(2, 8))).astype(np.float32)
    m = gaussian_filter(m, sigma=rng.uniform(1, 4))
    return (m > 0.3).astype(np.float32)

# %% [markdown]
# ## Dataset

# %%
class SSLData(Dataset):
    def __init__(self, files, seed=0):
        self.files = files
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        rgb = np.array(Image.open(self.files[i]).convert("RGB").resize((SIZE, SIZE)), np.uint8)
        y = weak_label(rgb)                                  # (H,W) clean weak mask
        prior = corrupt(y, self.rng)                         # (H,W) sloppy prior
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = np.concatenate([x.transpose(2, 0, 1), prior[None]], 0)   # (4,H,W)
        return torch.from_numpy(x).float(), torch.from_numpy(y[None]).float()

files = sorted(TILES_DIR.glob("*.png"))
print(f"{len(files)} unlabeled tiles for SSL")
dl = DataLoader(SSLData(files, SEED), batch_size=BATCH, shuffle=True,
                num_workers=2, drop_last=True)

# %% [markdown]
# ## Model (4-channel: RGB + prior) + loss

# %%
def build_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                    in_channels=4, classes=1).to(DEVICE)

def dice_focal(logits, target, alpha=0.25, gamma=2.0, smooth=1.0):
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    focal = (alpha * (1 - torch.exp(-bce)) ** gamma * bce).mean()
    inter = (prob * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + smooth) / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + smooth)
    return focal + dice.mean()

model = build_model()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# %% [markdown]
# ## Train

# %%
model.train()
for ep in range(1, EPOCHS + 1):
    tot = 0.0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = dice_focal(model(x), y); loss.backward(); opt.step()
        tot += loss.item() * x.size(0)
    sched.step()
    print(f"ep {ep:>2}/{EPOCHS}  loss={tot/len(dl.dataset):.4f}")
torch.save(model.state_dict(), OUT / "ssl_pretrained.pt")
print("saved", OUT / "ssl_pretrained.pt")

# %% [markdown]
# ## Visual check: image | weak label | corrupted prior | recovered

# %%
model.eval()
ds = SSLData(files, seed=123)
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
cols = ["image", "weak label (target)", "corrupted prior (input ch4)", "recovered"]
with torch.no_grad():
    for r in range(4):
        x, y = ds[r]
        prob = torch.sigmoid(model(x.unsqueeze(0).to(DEVICE)))[0, 0].cpu().numpy()
        rgb = (x[:3].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
        panels = [rgb, y[0].numpy(), x[3].numpy(), prob]
        for c, ax in enumerate(axes[r]):
            ax.imshow(panels[c], cmap=None if c == 0 else "magma", vmin=0, vmax=1)
            if r == 0: ax.set_title(cols[c], fontsize=9, fontweight="bold")
            ax.axis("off")
fig.suptitle("SSL corrupt→recover (unlabeled 30cm)", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "ssl_check.png", dpi=150); plt.show()
print("Next: finetune_30cm.py can init from ssl_pretrained.pt")
