"""
analyze_results_by_class.py — divide-and-conquer ANALYSIS (not separate models).
================================================================================
Run this AFTER finetune_30cm.py. It joins the per-tile out-of-fold IoU
(finetune30cm_outputs/finetune30cm_metrics.csv) with index.csv (site, severity,
species/DCA) and reports where the ONE model works and where it fails. This is the
right way to use our diversity: keep a single pooled model (small data → can't split),
but SLICE the results so we see which sites/species/severities need more data next.

    python analyze_results_by_class.py
"""
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("data/seed30cm")
MET = DATA / "finetune30cm_outputs" / "finetune30cm_metrics.csv"
if not MET.exists():
    raise SystemExit(f"{MET} not found — run finetune_30cm.py first (it writes the per-tile IoU).")

met = pd.read_csv(MET, dtype={"id": str}); met["id"] = met["id"].str.zfill(4)
idx = pd.read_csv(DATA / "index.csv", dtype={"id": str}); idx["id"] = idx["id"].str.zfill(4)
df = met.merge(idx[["id", "area_cluster", "pct_affected", "dca", "host", "role"]], on="id", how="left")

# Name the HEADLINE columns explicitly. The old code looked for "iou_with_prior"/"recall_with_prior"
# (names finetune_30cm.py never wrote) and silently fell back to "the last column containing iou" —
# which depended on column ORDER and would grab the with-prior ablation column instead of the
# headline no-prior one. 2026-07-22: be explicit, and fail loudly rather than report the wrong arm.
metrics = [c for c in ("iou_no_prior", "recall_no_prior") if c in df.columns]
if not metrics:      # pre-2026-07-22 CSVs used the *_soft_prior names
    metrics = [c for c in ("iou_soft_prior", "recall_soft_prior") if c in df.columns]
    if metrics:
        print("NOTE: this is an OLD metrics.csv — reporting the WITH-prior arm, not the headline.")
if not metrics:
    raise SystemExit(f"no known IoU column in {MET}; columns are {list(df.columns)}")
print(f"metrics = {metrics}   (out-of-fold; NaN = no-damage tile, excluded)\n")

def slice_report(by, name):
    print(f"=== by {name} (worst IoU first — these need more data / attention) ===")
    agg = {"n": (metrics[0], "size")}
    for m in metrics:
        agg[m] = (m, lambda s: np.nanmean(s))
    g = df.groupby(by).agg(**agg).sort_values(metrics[0])
    print(g.round(3).to_string()); print()

slice_report("area_cluster", "SITE")
slice_report("pct_affected", "SEVERITY")
slice_report("dca", "SPECIES / DISEASE")
for m in metrics:
    print(f"OVERALL {m}: mean={np.nanmean(df[m]):.3f}  median={np.nanmedian(df[m]):.3f}")
print("Read: any site/species far below the overall mean = the model's weak spot -> add labeled tiles there next.")
