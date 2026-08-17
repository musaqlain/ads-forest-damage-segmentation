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
| **Same, 60-epoch budget** (`-ep60`) | **0.290 ± 0.013** *(n=3)* | 0.50 | 8.8% |

Per-source-tile IoU (each annotated site gets one vote, rather than one vote per crop) tracks the
per-crop number closely — 0.296 and 0.266 against per-crop 0.299 and 0.275 — confirming the headline is
not carried by a few large sites.

**The `-ep60` row is honest but not significant.** Runs: 0.2954, 0.2992, **0.2748**. The third landed
0.02 below the first two, so the group now overlaps the 30-epoch group (whose best run was 0.2777) and
the permutation test gives **p = 0.057** at n=3 vs 4 — down from the p=0.029 that a third run *near the
first two* would have produced. The effect is retained because it is consistent in direction across
three runs and has an identified mechanism (evidence chain 6), not because it clears 0.05.

**Provenance.** Only two of the three `-ep60` runs reached `run_history.csv`; the first crashed in the
reporting cell (an uninitialised `_rows_mpp` on the path where the resolution diagnostic is skipped —
since fixed, and predictions are now cached right after CV so a late crash can never cost a run again).
Its cross-validation completed and printed 0.2954, which is why n=3 is quoted here while the CSV
stability line says n=2.

**Note on the two-run version of this table.** An earlier revision quoted `0.297 ± 0.003 (n=2)` and
called it provisional. That spread was an artefact of two runs happening to agree; the true spread is
~5x larger. Recorded here as a caution: n=2 does not estimate a standard deviation.

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
ceiling — and it did, in 2 of 5 folds. At 60 epochs: IoU 0.257 → **0.290 ± 0.013**, precision 0.50 →
0.54, and **0 of 5 folds** select the last epoch in any of the three runs, so 60 is sufficient rather
than merely larger. Inner-validation IoU rose in every fold of every run (+0.02 to +0.05), which is the
mechanism rather than the outcome, measured fifteen independent times. The script now warns explicitly
when folds cluster at the ceiling, so this cannot recur silently. Effect size is modest and p = 0.057;
see the headline note.

**7. Selective deployment is unnecessary — the model beats the survey everywhere.** Ranking crops by
label-free confidence and emitting corrections only for the top *k*% is never better than correcting
all of them. Two runs, system IoU at 10/25/50/75/100%: `0.153 0.200 0.268 0.297 | 0.299` and
`0.146 0.187 0.243 0.278 | 0.275`. Withholding means keeping a polygon worth 0.115, and even the
model's weakest crops beat that. Confidence still correlates with quality (rho = +0.62 in both runs for
peak probability) and is useful for ordering human review, but part of that is mechanical — larger
patches raise both confidence and IoU — so it ranks well without demonstrating error awareness.

**8. Both failure modes come from one learned rule: bare ground means damage.** Measured with an
excess-green openness proxy (`2G − R − B`) per crop, over the two runs that ran the diagnostic:

| test | run 1 | run 2 | reading |
|---|---|---|---|
| openness vs IoU, damage crops | rho = −0.013 (p = 0.56) | rho = +0.063 (p = 0.006) | **null** — noise either side of zero |
| openness, silent vs firing crops | 0.45 vs 0.52 (p < 0.0001) | 0.44 vs 0.53 (p < 0.0001) | **replicated** — silent crops are in **closed** canopy |
| false-alarm area, 412 verified-healthy crops | 15.5% open vs 1.2% closed, rho = +0.464 | 13.2% vs 5.1%, rho = +0.337 | **replicated in direction** (both p < 0.0001); magnitude is not stable |

Recall rises monotonically with openness in both runs (0.458 → 0.633 and 0.408 → 0.611) while the
silent share falls by half to two thirds (17.4% → 8.0%, 23.1% → 6.7%). One rule, two opposite symptoms:
over-prediction on bare ground, silence under closed canopy. Row 3 is the decisive one because those
crops are confirmed damage-free, so no partial-label explanation is available — but quote it as "several
times worse", not as a fixed multiple. The *open* value is the stable half (15.5%, 13.2%); the
closed-canopy baseline moves between runs, which is what drives the ratio from 13x down to 2.6x.

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

Latest single run; averaging cells across runs would hide which run each came from. The *shape* is what
replicates.

| damage in crop | n | IoU | recall | silent |
|---|---|---|---|---|
| <0.5% | 141 | 0.019 | 0.19 | 38% |
| 0.5–1% | 117 | 0.100 | 0.50 | 26% |
| 1–2% | 217 | 0.161 | 0.51 | 26% |
| 2–5% | 357 | 0.235 | 0.57 | 17% |
| 5–10% | 306 | 0.331 | 0.52 | 14% |
| 10–25% | 397 | 0.363 | 0.50 | 10% |
| >25% | 373 | 0.390 | 0.46 | 4% |

IoU rises 20× while recall stays flat above 0.5%. The gradient is mostly the geometry of IoU — a
fixed boundary error costs far more on a small target — not the model failing. The exception is the
bottom row, where recall genuinely collapses to 0.19 and 38% of crops go silent. Median
predicted/label area ratio is **0.9×**: the model predicts roughly the right *amount* of damage in the
wrong *place*, so this is a localisation problem, not a calibration one.

---

## Known limitations

1. **Small damage is missed** — ~14% of true damage crops get no prediction at all (253 and 300 in the
   two latest runs). A silent crop scores recall 0, and silent-count tracks the run-to-run recall swing
   almost exactly. The model has learned a large-area texture cue and cannot resolve individual dead
   crowns.
2. **Commission ~8.8%** on verified damage-free tiles, up from ~2% on whole tiles, because cropping cut
   the share of confirmed negatives from 29% to 18%. Worst crop 94–99% painted. Two untried fixes, in
   cost order: overlap the negative crops (`STRIDE_FRAC_NEG < 1.0`, no annotation at all), then collect
   confirmed negatives in open woodland — which evidence chain 8 identifies as the specific gap. The
   openness proxy can select candidate tiles automatically, and negatives need verification rather
   than tracing, so this is far cheaper per tile than damage annotation.
3. **Partial labels** — annotators tightened and split ADS blobs but did not exhaustively trace every
   dead tree, so IoU is a lower bound.
4. **146 labelled tiles.** Volume is a constraint, but evidence chain 8 says *which* data, and the
   answer is not "more of the same".
5. **Fold 2 holds 42% of the test weight** and is usually the weakest fold, which is most of the
   run-to-run spread: across the three `-ep60` runs fold 2 scored 0.302, 0.265, 0.243 while fold 4
   scored 0.296, 0.362, 0.309. The tile-weighted headline therefore sits below the unweighted fold mean
   (0.2748 vs 0.2825 latest). A single fold carrying 42% of the weight is the main structural weakness
   of the evaluation.
6. **Detection is only fair and unstable.** ROC-AUC 0.754 then 0.703 across two runs, PR-AUC 0.895 then
   0.881 against a 0.822 random baseline — the model localises better than it decides whether a crop is
   damaged at all, and this metric swings more between runs than IoU does.
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
