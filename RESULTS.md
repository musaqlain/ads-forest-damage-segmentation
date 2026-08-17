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
| Fine-tuned, fixed 0.60 m/px crops, 30-epoch budget | 0.257 ± 0.020 *(n=4)* | 0.47 ± 0.04 | 7.0% ± 2.5% |
| **Same, 60-epoch budget** (`-ep60`) | **0.297 ± 0.003** *(n=2)* | **0.51** | 8.4% |

Per-source-tile IoU (each annotated site gets one vote, rather than one vote per crop) is **0.296** on
the best run — level with the per-crop 0.299, confirming the headline is not carried by a few large
sites.

**Provenance of the `-ep60` n=2.** Both runs completed cross-validation; only the second reached
`run_history.csv`, because the first crashed in the reporting cell (an uninitialised `_rows_mpp` when
the resolution diagnostic is skipped — since fixed, and predictions are now cached to disk right after
CV so this can never cost a run again). The two tile-weighted means were 0.2954 and 0.2992. A third
run is pending; until it lands this row is provisional, and the CSV will show n=1 for it.

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

**6. The score was capped by the epoch budget, not the method.** `FT_EPOCHS = 30` was chosen so a crop
run would not take longer than the whole-tile recipe it had to be compared against. But the shipped
epoch is *selected* on inner-validation, so the budget only binds if the selection presses against the
ceiling — and it did, in 2 of 5 folds. At 60 epochs: IoU 0.257 → **0.297**, precision 0.50 → 0.55,
silent crops 290 → 253, and **0 of 5 folds** now select the last epoch, so 60 is sufficient rather
than merely larger. Inner-validation IoU rose in all five folds (+0.02 to +0.05), which is the
mechanism rather than the outcome, measured five independent times. The script now warns explicitly
when folds cluster at the ceiling, so this cannot recur silently.

**7. Selective deployment is unnecessary — the model beats the survey everywhere.** Ranking crops by
label-free confidence and emitting corrections only for the top *k*% scores *worse* at every *k* than
correcting all of them (0.153 / 0.200 / 0.268 / 0.297 at 10/25/50/75% vs 0.299 at 100%). Withholding
means keeping a polygon worth 0.115, and even the model's weakest crops beat that. Confidence still
correlates with quality (rho = +0.62 for peak probability) and is useful for ordering human review,
but part of that is mechanical — larger patches raise both confidence and IoU — so it ranks well
without demonstrating error awareness.

**8. Both failure modes come from one learned rule: bare ground means damage.** Measured with an
excess-green openness proxy (`2G − R − B`) per crop:

| test | result | reading |
|---|---|---|
| openness vs IoU, damage crops | rho = −0.013, p = 0.56 | **null** — openness does not predict IoU |
| openness, silent vs firing crops | 0.45 vs 0.52, p < 0.0001 | silent crops are in **closed** canopy |
| openness vs false-alarm area, 412 verified-healthy crops | rho = +0.464, p < 0.0001 | **15.5% open vs 1.2% closed — 13x** |

Recall rises monotonically with openness (0.458 → 0.633) while the silent share halves (17.4% → 8.0%).
One rule, two opposite symptoms: over-prediction on bare ground, silence under closed canopy. The
third row is the decisive one because those crops are confirmed damage-free, so no partial-label
explanation is available.

This **corrects an earlier claim** in this file and the README that both failure modes occurred in open
woodland. That was inferred from inspecting contact sheets; rows 1 and 2 above contradict it. Retained
here rather than deleted, because the error is instructive: three dozen thumbnails looked conclusive
and were not.

---

## Negative and null results

Worth reporting; each one cost time and closed a direction.

| Tried | Outcome |
|---|---|
| Affine alignment (ProximityAlign-style) | Fails — the polygon shape is wrong, not its position |
| NDVI / NIR as an extra input channel | Hurt, then no-op once the `conv1` re-init bug was fixed |
| ADS prior as a soft distance-transform hint | Consistently negative across 5 logged runs |
| Per-fold threshold tuning | **Reversed 2026-08-09**: the paired A/B now gives −0.024 IoU for fixed 0.5 (0.278 vs 0.254) and 415 silent crops vs 290. Tuning helps; earlier fixed-0.5 numbers were the optimistic ones |
| Averaging seed repeats at a fixed 0.5 threshold | Makes variance *worse* — averaging sharpens probabilities |
| Raising the minimum-label threshold to drop thin crops | Would lift IoU 0.299 → 0.415 while leaving recall flat — a metric artefact, not an improvement. Rejected |
| Gating corrections on model confidence before deployment | Worse at every coverage than correcting everything (see evidence chain 7). Rejected |
| Restricting the epoch budget to match the whole-tile recipe | Cost 0.04 IoU for two months (see evidence chain 6) |

### Performance by damage size

| damage in crop | n | IoU | recall | silent |
|---|---|---|---|---|
| <0.5% | 141 | 0.022 | 0.19 | 38% |
| 0.5–1% | 117 | 0.122 | 0.47 | 31% |
| 1–2% | 217 | 0.183 | 0.54 | 21% |
| 2–5% | 357 | 0.254 | 0.59 | 12% |
| 5–10% | 306 | 0.368 | 0.57 | 11% |
| 10–25% | 397 | 0.393 | 0.54 | 8% |
| >25% | 373 | 0.415 | 0.49 | 3% |

IoU rises 19× while recall stays flat above 0.5%. The gradient is mostly the geometry of IoU — a
fixed boundary error costs far more on a small target — not the model failing. The exception is the
bottom row, where recall genuinely collapses to 0.19 and 38% of crops go silent. Median
predicted/label area ratio is **0.9×**: the model predicts roughly the right *amount* of damage in the
wrong *place*, so this is a localisation problem, not a calibration one.

---

## Known limitations

1. **Small damage is missed** — 13% of true damage crops get no prediction at all (253 of 1,908 in
   the latest run, down from 290). A silent crop scores recall 0, and silent-count tracks the
   run-to-run recall swing almost exactly. The model has learned a large-area texture cue and cannot
   resolve individual dead crowns.
2. **Commission 8.4%** on verified damage-free tiles, up from ~2% on whole tiles, because cropping cut
   the share of confirmed negatives from 29% to 18%. Worst crop 94% painted. Two untried fixes, in
   cost order: overlap the negative crops (`STRIDE_FRAC_NEG < 1.0`, no annotation at all), then collect
   confirmed negatives in open woodland — which evidence chain 8 identifies as the specific gap. The
   openness proxy can select candidate tiles automatically, and negatives need verification rather
   than tracing, so this is far cheaper per tile than damage annotation.
3. **Partial labels** — annotators tightened and split ADS blobs but did not exhaustively trace every
   dead tree, so IoU is a lower bound.
4. **146 labelled tiles.** Volume is a constraint, but evidence chain 8 says *which* data, and the
   answer is not "more of the same".
5. **Fold 2 holds 42% of the test weight** and is usually the weakest fold. In the latest run the
   tile-weighted headline sits ~0.004 below the unweighted fold mean (0.2992 vs 0.3035).
6. **Detection is only fair.** ROC-AUC 0.754, PR-AUC 0.895 against a 0.822 random baseline — the
   model localises better than it decides whether a crop is damaged at all.
7. **No out-of-region test.** Cross-validation is spatially blocked but entirely within Oregon, so
   nothing here measures transfer to another state. One region should be held out completely and
   scored once, at the end.
8. **The `damage` label definition is unsettled.** Six of the 36 worst crops come from one site where
   a large bare clearing inside closed forest is labelled as damage and the model predicts nothing.
   Whether complete mortality that has become bare ground counts as damage is a question for the
   survey, not the model — and it directly conflicts with the rule the model must learn to avoid
   false alarms in open woodland.

---

## Reproducing

```bash
python build_crops_30cm.py      # 206 tiles -> 2,320 crops at a fixed 0.60 m/px
# then finetune_30cm.py on Colab; ~70 min per full 5-fold run on an A100
```

`FOLDS_TO_RUN = [0, 2]` screens a configuration ~3× faster by testing 2 of the 5 folds while still
training on 80% of the data. It tags `train_ver` with `-folds2` so partial runs can never pool with
reportable ones. Use `None` for anything you intend to report.
