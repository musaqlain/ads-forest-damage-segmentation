"""
build_annotation_queue.py  — stage ONLY the annotatable tiles for Labelme
=========================================================================
Labelme has no "only these files" filter: `labelme <dir>` shows EVERY image in <dir>.
So to annotate only the good tiles, we hand Labelme a folder that contains only them.

This copies the damage tiles worth tracing — severity Moderate or denser (>=11% dead
trees) — into `data/seed30cm/images_queue/`, then you point Labelme at that folder. It
SKIPS the Light (4-10%) tiles (a big polygon with only a sparse scatter of dead trees =
slow, noisy to trace) and the empty negatives (they already have zero masks).

Why a copy and not a move: `images/` stays the single source of truth. `labelme_to_masks.py`
reads each tile's pixel size from `images/<id>.png` and matches JSONs by id, so nothing about
the existing pipeline changes — the queue is just a disposable "to-do inbox".

Idempotent + resumable: it rebuilds the queue each run and skips tiles you've already annotated
(a JSON already exists in `data/seed30cm/labelme/`), so re-running always leaves exactly the
tiles still to do. Safe to run again after every annotation session.

Usage (run in ads_env, on the machine where you annotate):
    python build_annotation_queue.py
    labelme data/seed30cm/images_queue --output data/seed30cm/labelme --labels damage,ignore
    python labelme_to_masks.py            # unchanged — reads sizes from images/, JSONs from labelme/

Point SEED30CM_DIR at the 587-tile folder if it isn't ./data/seed30cm (e.g. your synced Drive path).
"""
import os
import json
import shutil
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("SEED30CM_DIR", "data/seed30cm"))
# Severity = DENSITY of dead trees, not polygon size. Moderate+ = an obvious brown patch you can
# trace; Light (4-10%) is sparse and slow. Edit this list if you later want to include Light.
KEEP = ["Very Severe (>50%)", "Severe (30-50%)", "Moderate (11-29%)"]
# Prioritise the model's WEAK classes first (from analyze_results_by_class.py): bark beetles sit well
# BELOW the overall IoU and are the bulk of the labelled data, so they're the highest-value tiles to add
# next. Batch 1: leave this list as-is (queues ONLY these species). Batch 2: set FOCUS_DCA = [] to queue
# everything else for diversity. Edit the species names to match your index.csv `dca` values exactly.
FOCUS_DCA = ["mountain pine beetle", "Douglas-fir beetle", "western pine beetle", "balsam woolly adelgid"]

idx = pd.read_csv(DATA / "index.csv")
if "role" not in idx.columns:
    idx["role"] = "damage"                                  # older index.csv predates the role column
idx["id"] = idx["id"].astype(str).str.zfill(4)             # ids are 4-digit ("0087"); zfill undoes CSV int-cast
sel = idx[(idx["role"] == "damage") & idx["pct_affected"].isin(KEEP)].copy()
if FOCUS_DCA and "dca" in sel.columns:                      # weak-class-first: queue ONLY these species this batch
    n_all = len(sel); sel = sel[sel["dca"].isin(FOCUS_DCA)]
    print(f"FOCUS_DCA on -> queuing only weak classes: {len(sel)}/{n_all} Moderate+ tiles")
    print("  (set FOCUS_DCA = [] for your NEXT batch to queue everything else for diversity)")
dca_by_id = dict(zip(sel["id"], sel["dca"])) if "dca" in sel.columns else {}
good_ids = sel["id"].tolist()

queue = DATA / "images_queue"
queue.mkdir(parents=True, exist_ok=True)
for old in queue.glob("*.png"):                             # rebuild fresh so the queue = what's still to do
    old.unlink()

label_dir = DATA / "labelme"
def _human_reviewed(jp):                                    # a JSON still at prelabel's DRAFT_VERSION ("5.5.0")
    try:                                                    # is only a MACHINE auto-draft -> NOT done, re-queue.
        return str(json.loads(jp.read_text(encoding="utf-8")).get("version", "")) != "5.5.0"
    except Exception:
        return False
done = {p.stem for p in label_dir.glob("*.json") if _human_reviewed(p)} if label_dir.exists() else set()

queued_ids = []; already = missing = 0
for tid in good_ids:
    if tid in done:                                         # annotated in a previous session — leave it out
        already += 1
        continue
    src = DATA / "images" / f"{tid}.png"
    if not src.exists():
        missing += 1
        continue
    shutil.copy2(src, queue / f"{tid}.png")
    queued_ids.append(tid)

print(f"good (Moderate+) tiles: {len(good_ids)}")
print(f"  queued to annotate : {len(queued_ids)}   -> {queue}")
print(f"  already annotated  : {already}")
if missing:
    print(f"  image not found    : {missing}  (is SEED30CM_DIR the 587-tile folder?)")
if dca_by_id and queued_ids:                               # show WHAT you're about to annotate, by species
    counts = pd.Series([dca_by_id.get(t, "?") for t in queued_ids]).value_counts()
    print("  queued by species:")
    for name, c in counts.items():
        print(f"    {c:>4}  {name}")
print("\nNext:")
print(f'  labelme "{queue}" --output "{label_dir}" --labels damage')
print("  python labelme_to_masks.py")
