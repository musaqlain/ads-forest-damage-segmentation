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
# # Ben Step 3–4 — can SHEAR and SCALE be recovered? (translation + rotation already done)
#
# Ben's step 4 asks: *for each perturbation type in isolation, can an AffineRegistrationNet
# recover it, and up to what magnitude?* You already answered **translation** and **rotation**
# (see `explore_perturbation_recovery.py`). This notebook runs the two remaining types —
# **scale** and **shear (skew)** — so the step-4 table is complete.
#
# How it works: for each magnitude we train a *specialist* net on ONLY that one perturbation,
# then measure how far a transformed boundary lands from its target (**corner error, in px**)
# versus the **identity baseline** (= leave the polygon where it is). Recovered ⇒ the model's
# error sits far below the baseline. We also report a **decomposed** error you can quote directly:
# scale factor error (fraction) and shear-angle error (degrees).
#
# Verdict rule: `RECOVERED` if corner error < 40% of the identity baseline; `partial` if it
# closes >30% of the gap; else `NOT recovered` (the net learned "do nothing").
#
# Needs `explore_perturbation_recovery.py`, `generate_simulated_pairs.py`, `augmentation.py`
# in the same folder (they are). Runs in ~2–5 min on a GPU.

# %%
from explore_perturbation_recovery import REGIMES, run_one, plot_curves, print_summary, FULL, QUICK

QUICK_MODE = False       # False = FULL (400 steps/point — use this on the A100). True = fast CPU smoke test.
CFG = QUICK if QUICK_MODE else FULL
SUBSET = {k: REGIMES[k] for k in ["scale (+/- frac)", "shear (deg)"]}

# %%
results = {}
for name, spec in SUBSET.items():
    decomp = "scale_err" if "scale" in name else "shear_err"       # the interpretable per-type error
    unit = "" if "scale" in name else " deg"
    print(f"\n### Regime: {name}   (mode={'QUICK' if QUICK_MODE else 'FULL'}, {CFG['n_steps']} steps/point)")
    rows = []
    for mag in spec["mags"]:
        r = run_one(spec["make"], mag, CFG)
        gap = 1.0 - r["corner_px"] / max(r["corner_px_baseline"], 1e-9)
        verdict = ("RECOVERED" if r["corner_px"] < 0.4 * r["corner_px_baseline"]
                   else "partial" if gap > 0.3 else "NOT recovered")
        print(f"  mag={mag:>6}: corner={r['corner_px']:6.2f}px  baseline={r['corner_px_baseline']:6.2f}px  "
              f"IoU={r['iou_recovered']:.2f}  {decomp}={r[decomp]:.3f}{unit}  gap={gap*100:4.0f}%  [{verdict}]")
        rows.append(r)
    results[name] = rows

# %%
plot_curves(results, "step4_scale_shear.png")
print_summary(results)
print("\nRead-out: a curve far BELOW the grey baseline = that perturbation is recoverable in isolation.")
print("Quote the decomposed error (scale fraction / shear degrees) alongside the corner error for Ben.")
