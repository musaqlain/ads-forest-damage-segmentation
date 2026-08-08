# Recovering Forest Damage Annotations from Aerial Imagery

Turning coarse U.S. Forest Service **Aerial Detection Survey (ADS)** sketch polygons into
pixel-accurate forest-damage regions in high-resolution aerial imagery.

**Google Summer of Code 2026** · [DeepForest](https://github.com/weecology/DeepForest) / [Weecology](https://github.com/weecology), University of Florida
Contributor: **Muhammad Saqlain** · Mentors: **Ben Weinstein**, **Josh Veitch-Michaelis**

---

## The problem

ADS polygons are drawn by surveyors from a moving aircraft. They mark roughly *where* a forest
damage outbreak is, but their boundaries do not follow the actual damage. They are the largest
existing record of U.S. forest health — and they are too coarse to train or evaluate a modern
computer-vision model against.

The original project plan assumed the polygons were **misaligned** and could be fixed by shifting
them. Testing that assumption was the first result of this project, and it was negative:

> **The corrections are reshapes, not shifts.** No affine transform (translation, rotation, scale,
> shear) recovers the true damage boundary, because the polygon's *shape* is wrong, not its position.

That finding redirected the project from **geometric alignment** to **semantic segmentation**:
predict the damage region directly from imagery, using the ADS polygon only as a weak hint.

## The result

A U-Net (ResNet-34 encoder) pretrained on [TreeFinder](https://proceedings.neurips.cc/paper_files/paper/2025/file/f22625283cf5812f45933610314259be-Paper-Datasets_and_Benchmarks_Track.pdf)
and fine-tuned on 146 hand-annotated 30 cm tiles from Oregon.

| | region IoU | recall |
|---|---|---|
| ADS polygon as-is (do-nothing baseline) | 0.115 | — |
| Pretrained model, zero-shot | 0.108 | — |
| **Fine-tuned, whole tiles resized** | 0.116 ± 0.027 | 0.33 |
| **Fine-tuned, fixed 0.60 m/px crops** | **0.251 ± 0.019** | **0.46** |

*(mean ± sd over 3–5 independent cross-validation runs)*

### The main finding: pixel scale, not data volume, was the bottleneck

Seed tiles cover between 180 m and 1500 m of ground. Resizing them all to 384×384 px meant the
model saw effective resolutions from **0.61 to 3.68 m/px** — a 6.1× spread — while its pretrained
features were learned at a constant **0.60 m/px**. A dead conifer crown is ~16 px at 0.60 m/px and
~2.5 px at 3.7 m/px, where the texture that separates dead canopy from bare soil no longer exists.

Cutting each tile into fixed-resolution crops instead of resizing it **more than doubled IoU**, and
the cleanest evidence needs no statistics at all:

> **Zero-shot IoU rose from 0.040 to 0.108 (2.7×)** — the *same* pretrained weights with *no*
> fine-tuning, on the *same* ground. Only the pixel scale changed. Being deterministic, this
> measurement has zero run-to-run noise.

![How the crops are built](docs/figures/method_cropping.png)

Across three runs, every crop run scored higher than every whole-tile run (permutation p = 0.018).

### Honest limitations

- **Small damage is missed.** ~25% of true damage crops receive no prediction at all. The model
  has learned a large-area texture cue; individual dead crowns fall below what it can resolve.
- **Commission rose to 6.5%** on tiles verified as damage-free (from ~2% on whole tiles), because
  cropping reduced the share of confirmed negatives from 29% to 18% of the dataset.
- **Labels are partial by design.** Annotators tightened and split ADS blobs but did not
  exhaustively trace every dead tree, so IoU is a *lower bound* on true performance.
- **146 labelled tiles.** Small. Data volume is now the next binding constraint.

---

## How it is evaluated

Honest evaluation was the hardest engineering problem here, more so than the model.

```
206 source tiles (146 damage + 60 confirmed damage-free)
        |  cut into 384x384 crops at a fixed 0.60 m/px
        v
2,320 crops (1,908 damage + 412 negative)
        |  K-Means on tile latitude/longitude
        v
5 spatially-blocked folds -> train on 4, test on the 5th, five times
```

Three rules make the number trustworthy:

1. **Spatial blocking.** Outbreaks cluster geographically and neighbouring tiles look nearly
   identical. A random split leaks, and inflates the score.
2. **Grouping by source tile.** All crops cut from one tile stay in the same fold, so overlapping
   crops cannot straddle train and test.
3. **A nested inner-validation slice.** The training epoch and the decision threshold are *chosen*,
   not learned — 30 epochs × 8 thresholds = 240 candidates. Choosing them on the test fold inflates
   the reported score by ≈ 0.056 (measured by simulation). They are chosen on a 15% slice carved out
   of the *training* folds instead, so the test fold is scored exactly once.

Every configuration change is logged to `run_history.csv` with a `train_ver` tag, and any comparison
smaller than the measured run-to-run spread is reported as noise rather than a result.

---

## Repository layout

### Current pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Build tiles | `build_30cm_seed_tiles.py` | Fetch 30 cm OSIP imagery around 2024 ADS polygons; save RGB + prior + NIR |
| | `build_diverse_seed_tiles.py` | Geographically balanced positives plus confirmed negatives |
| | `build_annotation_queue.py` | Order tiles for annotation |
| 2. Annotate | `labelme_seed.py` | Vectorise the ADS prior into a Labelme JSON to trace over |
| | `prelabel_regions.py` / `prelabel_learned.py` | Optional auto-draft (fixed threshold / learned from your labels) |
| | `labelme_to_masks.py` | Labelme JSON → binary masks |
| | `review_tiles.py` | Label-driven review pass over auto-built negatives |
| | `annotate_regions.py` | No-install matplotlib fallback annotator |
| | `dataset_manifest.py` | Track which tiles are reviewed and usable |
| 3. Pretrain | `colab_segmentation_treefinder.py` | U-Net on TreeFinder (15,489 tiles @ 0.60 m/px) |
| | `colab_ssl_pretrain.py` | Self-supervised corrupt→recover pretraining (exploratory) |
| 4. **Fix resolution** | **`build_crops_30cm.py`** | Cut tiles into fixed 0.60 m/px crops — *the main result* |
| 5. Train & evaluate | **`finetune_30cm.py`** | Spatially-blocked 5-fold CV, nested selection, paired A/B, run history |
| 6. Analyse | `analyze_results_by_class.py`, `search_logs.py` | Break results down by damage class; search run logs |

### Evidence from earlier stages

| Script | What it established |
|---|---|
| `colab_diagnose_real_pairs.py` | Corrections are reshapes, and the damage is visible in imagery |
| `generate_simulated_pairs.py`, `explore_perturbation_recovery.py`, `finish_scale_shear.py` | Which affine perturbations are recoverable at all |
| `colab_apply_to_monica.py` | Zero-shot transfer to 1 m 2009 imagery fails (domain gap) |
| `colab_finetune_monica.py` | Fine-tuning fixes it: region IoU 0 → 0.42 on 38 pairs |
| `augmentation.py`, `transforms.py`, `coarse_align.py` | Shared libraries (rasterisation, synthetic displacement, spectral stress map) |
| `build_seed_data.py`, `demo_weak_augmentation.py` | Proposal-era NAIP + DeepForest data preparation |

---

## Running it

```bash
pip install -r requirements.txt
```

Scripts are written in [jupytext](https://jupytext.readthedocs.io/) percent format, so each `.py`
is both a runnable script and a notebook. To open one in Colab:

```bash
jupytext --to ipynb finetune_30cm.py
```

The `build_*` and annotation scripts run locally against `data/`. The `colab_*` scripts and
`finetune_30cm.py` expect a Colab runtime with Google Drive mounted; both are configured at the top
of each file.

Typical order:

```bash
python build_30cm_seed_tiles.py      # fetch imagery around ADS polygons
python labelme_seed.py               # prepare tracing guides
labelme data/seed30cm/images --output data/seed30cm/labelme --labels damage,prior
python labelme_to_masks.py           # -> data/seed30cm/masks/
python build_crops_30cm.py           # -> fixed 0.60 m/px crops
# then run finetune_30cm.py on Colab (A100 recommended; ~70 min for a full 5-fold run)
```

`data/`, model checkpoints and generated figures are not tracked — every one of them is
reproducible from the scripts above.

---

## Status against the GSoC deliverables

| Expected outcome | Status |
|---|---|
| Approach for connecting airborne annotations and NAIP imagery | **Done** — core of the project |
| Weak dataset for tree-health model training from predicted locations | **Done** — ADS-prior seed tiles, 206 hand-reviewed |
| Computer-vision model for forest health outbreaks | **Done** — segmentation rather than detection, a deliberate change after the reshape finding |
| Improved DeepForest ↔ NAIP map-server connection | **Partial** — NAIP/OSIP tile fetching works; DeepForest not yet wired in |
| Blog post | In progress |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Mentors **Ben Weinstein** and **Josh Veitch-Michaelis** (Weecology, University of Florida).
Pretraining uses the **TreeFinder** dataset (Wang et al., NeurIPS 2025 Datasets & Benchmarks).
Imagery from USDA **NAIP** and Oregon **OSIP**; annotations from the USFS **Aerial Detection Survey**.
