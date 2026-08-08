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
# # Rebuild the 30cm seed tiles as fixed-resolution CROPS
#
# **Why.** The fine-tune initialises from `unet_treefinder_best.pt`, trained on 224x224 tiles at a
# constant **0.60 m/px**. The current pipeline resizes every seed tile to 384x384 regardless of the
# ground area it covers (180-1500 m), so the model sees **0.61-3.68 m/px** — a 6.1x spread, median
# 2.4x coarser than the encoder was pretrained at.
#
# Measured on the 2026-08-04 run, among the 91 tiles where the model actually fired:
# `rho(m/px, IoU) = -0.305, p = 0.003`. Finest bin IoU 0.155, coarse bins 0.058-0.066.
#
# **Mechanism.** A dead conifer crown and bare soil are the same colour; only texture and crown shape
# separate them. At 0.60 m/px a 9.5 m crown (TreeFinder's median) is ~16 px and clearly shaped. At
# 3.7 m/px it is ~2.5 px and the texture is gone. That is one cause behind two symptoms: the model
# paints bare ground as damage, AND goes silent inside dense canopy.
#
# **This notebook is READ-ONLY on your annotations.** It reads `images/ masks/ priors/ ignore/ nir/`
# and writes a NEW directory. Nothing in the source tree is modified, moved or deleted — it
# fingerprints the source folders before and after and prints the result.
#
# **On Colab it writes to local disk** (`/content/...`), not Drive. Writing thousands of small files
# to Drive is extremely slow; training from local disk is also much faster. At the end it saves ONE
# zip to Drive so the crops survive a session restart.

# %%
import hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    DATA_ROOT = Path("/content/drive/MyDrive/Data/seed30cm")
    OUT_ROOT  = Path("/content/seed30cm_crops")          # local disk: fast to write AND to train from
    ZIP_TO    = Path("/content/drive/MyDrive/Data/seed30cm_crops.zip")
else:
    DATA_ROOT = Path(__file__).resolve().parent / "data" / "seed30cm"
    OUT_ROOT  = DATA_ROOT.parent / "seed30cm_crops"
    ZIP_TO    = None

print("source :", DATA_ROOT)
print("output :", OUT_ROOT)

# %%
TARGET_GSD = 0.60      # m/px — exactly TreeFinder's pretraining resolution
CROP       = 384       # px per crop => 384 * 0.60 = 230 m of ground per crop
# Stride as a fraction of CROP. 1.0 = no overlap, 0.5 = 50% overlap.
# 0.5 measured (206 source tiles): 2320 crops kept = 1908 damage + 412 negative, after 2500 blank
# crops were dropped. That blank-drop is why the count stays modest -- with FT_EPOCHS=30 a full
# 5-fold CV is roughly 60-70 min, not the multi-hour run a naive count would suggest.
STRIDE_FRAC_POS = 0.5
STRIDE_FRAC_NEG = 1.0  # confirmed negatives are plentiful and easy; never worth overlapping

# A crop from a DAMAGE tile containing no traced label is NOT a safe negative — labelling is partial
# by design, so untraced damage may sit there. Dropping those is the conservative choice. Crops of a
# CONFIRMED-NEGATIVE tile are safe, because those tiles were verified damage-free as a whole.
MIN_LABEL_PX = 64          # a crop counts as positive at >= this many label px (~0.04% of 384^2)
NEG_RATIO    = 0.41        # negatives/positives, matching the current 60/146 so the variable under
                           # test stays RESOLUTION and not class balance
SEED         = 42

LAYERS = (("images", False), ("masks", True), ("priors", True), ("ignore", True), ("nir", True))
SRC = {k: DATA_ROOT / k for k, _ in LAYERS}


def sha1_of_tree(d: Path) -> str:
    """Fingerprint a directory so we can prove the source annotations were not touched."""
    if not d.exists():
        return "absent"
    h = hashlib.sha1()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.name.encode()); h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def full_px_for(win_m: float) -> int:
    """Pixels this tile's ground extent occupies at TARGET_GSD, floored at one crop.

    A tile covering less ground than one crop is upsampled to exactly CROP, giving an effective GSD
    of win_m/CROP which is FINER than target — the harmless direction.
    """
    return max(CROP, int(round(win_m / TARGET_GSD)))


def offsets_for(full_px: int, role: str):
    frac = STRIDE_FRAC_POS if role == "damage" else STRIDE_FRAC_NEG
    stride = max(1, int(round(CROP * frac)))
    offs = list(range(0, max(1, full_px - CROP + 1), stride))
    if full_px >= CROP and offs[-1] != full_px - CROP:
        offs.append(full_px - CROP)                      # always include the far edge
    return offs


# %%
assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
before = {k: sha1_of_tree(v) for k, v in SRC.items()}

idx = pd.read_csv(DATA_ROOT / "index.csv"); idx["id"] = idx["id"].astype(str).str.zfill(4)
man = pd.read_csv(DATA_ROOT / "manifest.csv"); man["id"] = man["id"].astype(str).str.zfill(4)
meta = (man[man["use"] == True][["id", "role"]]
        .merge(idx[["id", "window_m", "size_px", "lat", "lon"]], on="id", how="left")
        .dropna(subset=["window_m"]).reset_index(drop=True))
print(f"source tiles in use: {len(meta)}  "
      f"(damage {int((meta.role=='damage').sum())}, negative {int((meta.role=='negative').sum())})")

# %% [markdown]
# ## Pass 1 — decide which crops to keep (reads MASKS only, writes nothing)
# Cheap: single-channel masks, no RGB. Doing this first means pass 2 writes only the crops we keep,
# instead of writing everything and deleting the surplus.

# %%
t0 = time.time()
cand = []
for n, r in enumerate(meta.itertuples(index=False), 1):
    win_m, role = float(r.window_m), r.role
    fp = full_px_for(win_m)
    mp = SRC["masks"] / f"{r.id}.png"
    if mp.exists():
        m_full = np.array(Image.open(mp).convert("L").resize((fp, fp), Image.NEAREST)) > 128
    else:
        m_full = np.zeros((fp, fp), bool)
    offs = offsets_for(fp, role)
    for ri, y in enumerate(offs):
        for ci, x in enumerate(offs):
            npx = int(m_full[y:y + CROP, x:x + CROP].sum())
            if role == "damage" and npx < MIN_LABEL_PX:
                continue                                  # may hide untraced damage — do not use
            cand.append(dict(id=f"{r.id}_{ri}{ci}", source_id=r.id,
                             role="damage" if npx >= MIN_LABEL_PX else "negative",
                             label_px=npx, lat=float(r.lat), lon=float(r.lon),
                             y=y, x=x, full_px=fp, src_window_m=win_m,
                             window_m=CROP * win_m / fp, size_px=CROP,
                             gsd_m=round(win_m / fp, 4), crop_row=ri, crop_col=ci))
    if n % 50 == 0 or n == len(meta):
        print(f"  scanned {n}/{len(meta)} tiles -> {len(cand)} candidate crops "
              f"({time.time()-t0:.0f}s)", flush=True)

cand = pd.DataFrame(cand)
pos, neg = cand[cand.role == "damage"], cand[cand.role == "negative"]
want_neg = int(round(len(pos) * NEG_RATIO))
if len(neg) > want_neg:
    # Spread the sample across source tiles rather than taking a plain random draw, so geographic
    # coverage of the negatives is preserved.
    neg = (neg.sample(frac=1.0, random_state=SEED)
              .groupby("source_id", group_keys=False).head(max(1, want_neg // max(1, neg.source_id.nunique()) + 1))
              .head(want_neg))
keep = pd.concat([pos, neg]).sort_values(["source_id", "id"]).reset_index(drop=True)
print(f"\nkeeping {len(keep)} crops: {len(pos)} damage + {len(neg)} negative "
      f"(from {keep.source_id.nunique()} source tiles)")

# %% [markdown]
# ## Pass 2 — write the kept crops

# %%
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
for k, _ in LAYERS:
    (OUT_ROOT / k).mkdir(parents=True, exist_ok=True)

t0 = time.time()
for n, (sid, grp) in enumerate(keep.groupby("source_id"), 1):
    fp = int(grp.full_px.iloc[0])
    resized = {}
    for key, is_mask in LAYERS:
        p = SRC[key] / f"{sid}.png"
        if p.exists():
            im = Image.open(p).convert("L" if is_mask else "RGB")
        elif key in ("masks", "priors", "ignore"):
            im = Image.new("L", (fp, fp), 0)              # absent layer => empty
        else:
            continue                                      # nir is genuinely optional
        resized[key] = im.resize((fp, fp), Image.NEAREST if is_mask else Image.BILINEAR)
    for c in grp.itertuples(index=False):
        box = (c.x, c.y, c.x + CROP, c.y + CROP)
        for key, im in resized.items():
            im.crop(box).save(OUT_ROOT / key / f"{c.id}.png")
    if n % 25 == 0 or n == keep.source_id.nunique():
        print(f"  wrote {n}/{keep.source_id.nunique()} source tiles ({time.time()-t0:.0f}s)", flush=True)

# %%
out = keep.drop(columns=["y", "x", "full_px"])
out.to_csv(OUT_ROOT / "index.csv", index=False)
out.assign(bucket="crop", has_mask=out.label_px > 0, damage=out.label_px > 0, ignore=False, use=True)[
    ["id", "bucket", "has_mask", "damage", "ignore", "role", "use", "source_id"]
].to_csv(OUT_ROOT / "manifest.csv", index=False)

after = {k: sha1_of_tree(v) for k, v in SRC.items()}
untouched = all(before[k] == after[k] for k in before)

summary = dict(
    target_gsd=TARGET_GSD, crop_px=CROP, ground_per_crop_m=round(CROP * TARGET_GSD, 1),
    source_tiles=int(len(meta)), crops_total=int(len(out)),
    crops_damage=int((out.role == "damage").sum()),
    crops_negative=int((out.role == "negative").sum()),
    distinct_source_tiles=int(out.source_id.nunique()),
    gsd_min=float(out.gsd_m.min()), gsd_median=float(out.gsd_m.median()), gsd_max=float(out.gsd_m.max()),
    source_annotations_unmodified=bool(untouched))
(OUT_ROOT / "build_summary.json").write_text(json.dumps(summary, indent=2))

print("\n=== CROP BUILD SUMMARY ===")
for k, v in summary.items():
    print(f"  {k:34s} {v}")
print(f"\n  effective GSD across crops: {out.gsd_m.min():.3f} - {out.gsd_m.max():.3f} m/px "
      f"(was 0.61-3.68, a 6.1x spread)")
print(f"  label coverage on damage crops: mean "
      f"{100*out[out.role=='damage'].label_px.mean()/CROP**2:.2f}% of pixels")
print(f"\n  SOURCE ANNOTATIONS UNMODIFIED: {untouched}")

# %% [markdown]
# ## Save one zip to Drive
# So the crops survive a runtime restart. One large file transfers far faster than thousands of
# small ones. To restore in a later session:
# `!unzip -q /content/drive/MyDrive/Data/seed30cm_crops.zip -d /content/`

# %%
if IN_COLAB and ZIP_TO is not None:
    t0 = time.time()
    shutil.make_archive(str(ZIP_TO.with_suffix("")), "zip", root_dir=OUT_ROOT.parent,
                        base_dir=OUT_ROOT.name)
    print(f"saved {ZIP_TO}  ({ZIP_TO.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)")
    print("\nNEXT: in finetune_30cm, set USE_CROPS = True and DATA_ROOT = Path('/content')")
else:
    print("(local run — no zip step)")

# %% [markdown]
# ## Note on cross-validation
# Every crop carries `source_id`. Crops from ONE source tile MUST stay in the same CV fold, or
# overlapping near-duplicates leak between train and test and inflate the score. `finetune_30cm`
# already handles this: it clusters SOURCE TILES for the spatial folds and gives every crop its
# source's fold. Confirm the line `GROUPED_CV: clustering N SOURCE TILES (not M crops)` appears.
