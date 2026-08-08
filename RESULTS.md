# Results log

Every number below comes from a logged run in `run_history.csv`. Configurations are tagged with
`train_ver`; numbers from different tags are **not** comparable and are never pooled.

---

## Headline

| Setting | region IoU | recall | commission |
|---|---|---|---|
| ADS polygon as-is (`prior_echo`, no model) | 0.115 | — | — |
| Pretrained weights, zero-shot | 0.108 | — | — |
| Fine-tuned, whole tiles resized to 384 px | 0.116 ± 0.027 *(n=5)* | 0.33 | ~2% |
| **Fine-tuned, fixed 0.60 m/px crops** | **0.251 ± 0.019** *(n=3)* | **0.46** | 6.5% |

Per-source-tile IoU (each annotated site gets one vote, rather than one vote per crop) is **0.234**
— close to the per-crop 0.251, confirming the headline is not carried by a few large sites.

---

## Evidence chain

**1. The corrections are reshapes, not shifts.**
No affine transform recovers the true boundary. Raw-parameter MSE hides this; corner error and
decomposed rotation/scale metrics expose it. → the project moved to semantic segmentation.

**2. The damage is visible in the imagery.**
`colab_diagnose_real_pairs.py` — a human can trace the boundary from RGB alone, so the task is
learnable in principle.

**3. Pretraining on TreeFinder works at matched resolution.**
Dice 0.475, per-object recall 0.67, false-positive rate 0.2% on healthy tiles. The published
Mask2Former baseline on the same data is F1 0.519 ± 0.055 — **statistically indistinguishable**, and
obtained here under a *stricter* source-grouped split rather than the paper's random 80/20 split.

**4. Zero-shot transfer fails; fine-tuning fixes it.**
On 1 m 2009 imagery the pretrained model predicts almost nothing (IoU ≈ 0.009). Fine-tuning on 38
pairs takes region IoU to **0.426**, improving 38 of 38.

**5. Resolution, not data volume, was the bottleneck.** ← *main result*
Seed tiles span 180–1500 m of ground but were all resized to 384 px, giving the model 0.61–3.68 m/px
(a 6.1× spread) while its pretrained features were learned at a constant 0.60 m/px. Holding every
crop at 0.60 m/px instead:

- IoU **0.116 → 0.251**, recall **0.33 → 0.46**
- Every crop run beat every whole-tile run (permutation p = 0.018, at the floor for n = 3 vs 5)
- The 0.116 baseline is the *same training procedure* (`thr-tuned`). Against the best whole-tile
  procedure ever logged (`seeded-tta`, 0.191 ± 0.031, n = 3) crops still win outright — 0.251 vs
  0.191, again with no overlap between the two groups of runs (p = 0.050, the floor at n = 3 vs 3).
- **Zero-shot IoU 0.040 → 0.108** — same weights, no training, zero measurement noise. This is the
  cleanest single piece of evidence in the project.

---

## Negative and null results

Worth reporting; each one cost time and closed a direction.

| Tried | Outcome |
|---|---|
| Affine alignment (ProximityAlign-style) | Fails — the polygon shape is wrong, not its position |
| NDVI / NIR as an extra input channel | Hurt, then no-op once the `conv1` re-init bug was fixed |
| ADS prior as a soft distance-transform hint | Consistently negative across 5 logged runs |
| Per-fold threshold tuning | Worth ≈ 0.01, well inside run-to-run noise |
| Averaging seed repeats at a fixed 0.5 threshold | Makes variance *worse* — averaging sharpens probabilities |

---

## Known limitations

1. **Small damage is missed** — ~25% of true damage crops get no prediction. The model has learned a
   large-area texture cue and cannot resolve individual dead crowns.
2. **Commission 6.5%** on verified damage-free tiles, up from ~2%, because cropping cut the share of
   confirmed negatives from 29% to 18%. Untested fix: overlap the negative crops.
3. **Partial labels** — annotators tightened and split ADS blobs but did not exhaustively trace every
   dead tree, so IoU is a lower bound.
4. **146 labelled tiles.** Data volume is now the binding constraint.
5. **Fold 2 holds 42% of the test weight** and is consistently the weakest fold (0.218 vs 0.24–0.30),
   pulling the tile-weighted headline ~0.01 below the unweighted fold mean.

---

## Reproducing

```bash
python build_crops_30cm.py      # 206 tiles -> 2,320 crops at a fixed 0.60 m/px
# then finetune_30cm.py on Colab; ~70 min per full 5-fold run on an A100
```

`FOLDS_TO_RUN = [0, 2]` screens a configuration ~3× faster by testing 2 of the 5 folds while still
training on 80% of the data. It tags `train_ver` with `-folds2` so partial runs can never pool with
reportable ones. Use `None` for anything you intend to report.
