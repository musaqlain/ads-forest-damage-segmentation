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
| **Fine-tuned, fixed 0.60 m/px crops** | **0.257 ± 0.020** *(n=4)* | **0.47 ± 0.04** | 7.0% ± 2.5% |

Per-source-tile IoU (each annotated site gets one vote, rather than one vote per crop) is
**0.245 ± 0.021** — close to the per-crop 0.257, confirming the headline is not carried by a few
large sites.

The best single crop run reached 0.278 with recall 0.520. It is **not** quoted as the headline: one
of its five folds failed to converge, and picking the best of four runs is selection, not a result.

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

- IoU **0.116 → 0.257**, recall **0.33 → 0.47**
- Every crop run beat every whole-tile run (permutation p = 0.008, at the floor for n = 4 vs 5)
- The 0.116 baseline is the *same training procedure* (`thr-tuned`). Against the best whole-tile
  procedure ever logged (`seeded-tta`, 0.191 ± 0.031, n = 3) crops still win outright — 0.257 vs
  0.191, again with no overlap between the two groups of runs (p = 0.029, the floor at n = 4 vs 3).
- **Zero-shot IoU 0.040 → 0.108** — same weights, no training, zero measurement noise. This is the
  cleanest single piece of evidence in the project.

Note that the resolution diagnostic inside a *crop* run is uninformative by construction: every crop
is 0.60 m/px, so metres-per-pixel has no variance to correlate against and `rho` is ~0 with p ~ 1.
That is not a refutation of the finding — it is what fixing a variable looks like. The evidence is
the whole-tile correlation (rho = −0.305, p = 0.003) and the zero-shot jump. The script now skips the
block and says so rather than printing a reading that inverts the conclusion.

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

1. **Small damage is missed** — 15% of true damage crops get no prediction at all (290 of 1,908 in
   the latest run). A silent crop scores recall 0, and silent-count tracks the run-to-run recall
   swing almost exactly. The model has learned a large-area texture cue and cannot resolve
   individual dead crowns.
2. **Commission 7.0% ± 2.5%** on verified damage-free tiles, up from ~2%, because cropping cut the
   share of confirmed negatives from 29% to 18%. Worst case: one healthy crop 99.6% painted.
   Untested fix: overlap the negative crops (`STRIDE_FRAC_NEG < 1.0`).
3. **The epoch budget truncates training.** `FT_EPOCHS = 30`; in the latest run folds 2 and 5 both
   selected epoch 30 of 30 on inner-validation, i.e. the model had not stopped improving. The
   headline is a floor. Raising the budget is the cheapest untried improvement.
4. **Partial labels** — annotators tightened and split ADS blobs but did not exhaustively trace every
   dead tree, so IoU is a lower bound.
5. **146 labelled tiles.** Data volume is now the binding constraint.
6. **Fold 2 holds 42% of the test weight** and is consistently the weakest fold, pulling the
   tile-weighted headline ~0.01 below the unweighted fold mean (0.278 vs 0.280 latest).
7. **Detection is only fair.** ROC-AUC 0.745, PR-AUC 0.894 against a 0.822 random baseline — the
   model localises better than it decides whether a crop is damaged at all.

---

## Reproducing

```bash
python build_crops_30cm.py      # 206 tiles -> 2,320 crops at a fixed 0.60 m/px
# then finetune_30cm.py on Colab; ~70 min per full 5-fold run on an A100
```

`FOLDS_TO_RUN = [0, 2]` screens a configuration ~3× faster by testing 2 of the 5 folds while still
training on 80% of the data. It tags `train_ver` with `-folds2` so partial runs can never pool with
reportable ones. Use `None` for anything you intend to report.
