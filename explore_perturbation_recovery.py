"""
explore_perturbation_recovery.py
================================
Step 4 of the ADS polygon realignment roadmap:
  "Explore which types of perturbations can be recovered, and which cannot."

Experimental design
-------------------
For each perturbation TYPE in {translation, rotation, scale, shear} we train a
*specialist* AffineRegistrationNet on ONLY that perturbation, at a sweep of
MAGNITUDES, and measure how well it recovers the perturbation on a held-out set.

This isolates one degree of freedom at a time -- the cleanest way to answer
"is this recoverable in isolation, and up to what magnitude?". (A natural
follow-up, left for you to try, is a single *generalist* model trained on the
full mix, whose error you then break down by type -- see NOTES at the bottom.)

The headline, convention-free recovery metric is the corner / reprojection error
in pixels (mean distance a transformed boundary point lands from its target).
We compare it against the "identity baseline" = leaving the polygon where it is.

Verdict rule of thumb:
  recoverable   if  trained corner error  <  ~40% of the identity-baseline error
  not recovered if  trained corner error  ~=  baseline (network learned "do nothing")

Run:
  # quick local smoke test (CPU): coarse sweep, few steps
  python explore_perturbation_recovery.py --quick
  # full run (use a GPU / Colab):
  python explore_perturbation_recovery.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_simulated_pairs import (
    AffineRegistrationNet,
    SimulatedPairDataset,
    corner_error_px,
    evaluate,
    reference_grid_points,
    train,
    SEED,
    W,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================================
# Experiment definitions
# ===============================================================================
# Each regime maps a scalar "magnitude" m to the dataset_kwargs that isolate one
# perturbation type at that magnitude (everything else identity).

def kwargs_translation(m):  # m = max translation in pixels
    return dict(max_translation_px=m, max_rotation_deg=0.0,
                scale_range=(1.0, 1.0), max_shear_deg=0.0)

def kwargs_rotation(m):     # m = max rotation in degrees
    return dict(max_translation_px=0.0, max_rotation_deg=m,
                scale_range=(1.0, 1.0), max_shear_deg=0.0)

def kwargs_scale(m):        # m = +/- fractional scale (e.g. 0.2 -> 0.8..1.2)
    return dict(max_translation_px=0.0, max_rotation_deg=0.0,
                scale_range=(1.0 - m, 1.0 + m), max_shear_deg=0.0)

def kwargs_shear(m):        # m = max shear in degrees
    return dict(max_translation_px=0.0, max_rotation_deg=0.0,
                scale_range=(1.0, 1.0), max_shear_deg=m)


REGIMES = {
    "translation (px)": dict(make=kwargs_translation,
                             mags=[5, 10, 20, 30, 45, 60, 90]),
    "rotation (deg)":   dict(make=kwargs_rotation,
                             mags=[2, 5, 10, 20, 30, 45]),
    "scale (+/- frac)": dict(make=kwargs_scale,
                             mags=[0.05, 0.10, 0.20, 0.30, 0.40]),
    "shear (deg)":      dict(make=kwargs_shear,
                             mags=[2, 5, 10, 20, 30]),
}

QUICK = dict(n_steps=120, batch_size=32, n_eval=120)
FULL = dict(n_steps=400, batch_size=32, n_eval=250)


# ===============================================================================
# One experiment cell: train a specialist, evaluate it
# ===============================================================================

def run_one(make_kwargs, magnitude: float, cfg: dict) -> dict:
    dkw = make_kwargs(magnitude)
    torch.manual_seed(SEED)                       # reproducible model init
    model, _ = train(
        n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], lr=1e-3,
        loss_kind="reproj", log_every=10_000,     # quiet; we only want the summary
        device=DEVICE, dataset_kwargs=dkw,
    )
    m = evaluate(model, n_pairs=cfg["n_eval"], seed=999,
                 device=DEVICE, dataset_kwargs=dkw)
    m["magnitude"] = magnitude
    return m


# ===============================================================================
# Main sweep
# ===============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast CPU smoke test (fewer steps / eval pairs)")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL

    print("=" * 70)
    print("  STEP 4 — Which perturbations can be recovered?")
    print(f"  device={DEVICE}   mode={'QUICK' if args.quick else 'FULL'}   "
          f"steps/regime-point={cfg['n_steps']}")
    print("=" * 70)

    results = {}
    for name, spec in REGIMES.items():
        print(f"\n### Regime: {name}")
        rows = []
        for mag in spec["mags"]:
            r = run_one(spec["make"], mag, cfg)
            gap_closed = 1.0 - r["corner_px"] / max(r["corner_px_baseline"], 1e-9)
            verdict = ("RECOVERED" if r["corner_px"] < 0.4 * r["corner_px_baseline"]
                       else "partial" if gap_closed > 0.3 else "NOT recovered")
            print(f"  mag={mag:>6}:  corner={r['corner_px']:6.2f}px  "
                  f"baseline={r['corner_px_baseline']:6.2f}px  "
                  f"IoU={r['iou_recovered']:.2f}  "
                  f"gap_closed={gap_closed*100:4.0f}%   [{verdict}]")
            rows.append(r)
        results[name] = rows

    plot_curves(results, "step4_recoverability.png")
    print_summary(results)


def plot_curves(results: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, results.items()):
        mags = [r["magnitude"] for r in rows]
        corner = [r["corner_px"] for r in rows]
        base = [r["corner_px_baseline"] for r in rows]
        ax.plot(mags, base, "--", color="grey", label="identity baseline")
        ax.plot(mags, corner, "-o", color="#1f77b4", label="trained model")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("corner error (px)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Recoverability by perturbation type\n"
                 "(model curve far below baseline = recoverable; "
                 "model curve hugging baseline = not recovered)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  >  Recoverability curves saved to {out_path}")


def print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("  SUMMARY — recoverable magnitude range per perturbation type")
    print("=" * 70)
    for name, rows in results.items():
        recovered_mags = [r["magnitude"] for r in rows
                          if r["corner_px"] < 0.4 * r["corner_px_baseline"]]
        if recovered_mags:
            print(f"  {name:<18}: recovered up to ~{max(recovered_mags)}")
        else:
            best = min(rows, key=lambda r: r["corner_px"] / max(r["corner_px_baseline"], 1e-9))
            print(f"  {name:<18}: NOT cleanly recovered "
                  f"(best gap closed {(1-best['corner_px']/best['corner_px_baseline'])*100:.0f}%"
                  f" at mag={best['magnitude']})")
    print("=" * 70)


# ===============================================================================
# NOTES — things for YOU to explore next (write up what you find for Ben)
# ===============================================================================
# 1. Generalist vs specialist: train ONE model on the full mix (all 4 types at
#    once) and break its error down by type. Does jointly learning help or hurt
#    each component? This is closer to the real ADS task.
# 2. Architecture: the AdaptiveAvgPool2d((4,4)) is near-translation-invariant.
#    Try a finer pool, or add coordinate channels (CoordConv), and see whether
#    rotation/translation precision improves.
# 3. Loss ablation: rerun rotation with loss_kind="params" (raw-MSE). You should
#    see rotation fail even harder -- this demonstrates WHY the reprojection loss
#    matters and is a great figure for the blog post.
# 4. Symmetry: near-circular blobs are rotation-ambiguous. Try more elongated /
#    asymmetric shapes and see if rotation becomes more recoverable.

if __name__ == "__main__":
    main()
