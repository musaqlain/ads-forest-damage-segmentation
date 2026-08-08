"""
review_tiles.py — interactive Labelme review, driven entirely by LABELS you can see.
====================================================================================
No hidden checkboxes. When you draw a shape, Labelme pops up the label list; the label
you pick IS the verdict. Close the window and the script finishes the job (edits
dataset_manifest.py, rebuilds masks, rewrites manifest.csv).

Run it in ads_env (Labelme + OpenCV live there):

    python review_tiles.py negatives         # sweep the auto-built negative tiles
    python review_tiles.py empty             # review tiles opened but left empty
    python review_tiles.py empty 0451 0470   # review specific ids instead

THE THREE VERDICTS  (you pick a LABEL when you draw — that is all):
    * KEEP    -> draw NOTHING. Just go to the next tile (press D / the arrow).
                 A negative stays a NEGATIVE. This is the common case, zero effort.
    * DAMAGE  -> draw polygon(s) and pick label "damage".  The tile becomes a POSITIVE
                 (its mask is rasterised and it trains as damage).
    * REJECT  -> draw a quick box and pick label "reject".  The tile is DROPPED
                 (added to DELETE in dataset_manifest.py).
    (empty mode also offers "clean" -> confirmed NEGATIVE, added to NO_DAMAGE.)

Don't want to draw at all? Just note the bad ids by eye and run:

    python review_tiles.py drop 0455 0470 0483     # -> DELETE, no Labelme needed

Auto-save is ON, so there is nothing to press — review as many tiles as you like and
just CLOSE the window. Then confirm the printed summary and you're done.

Notes
* "reject"/"clean" are just markers — they never become mask pixels (stripped before
  anything reaches labelme_to_masks.py). "prior" (grey guide) is ignored too.
* Non-destructive: dataset_manifest.py is backed up before editing; only its DELETE /
  NO_DAMAGE lists are touched. Canonical labelme/*.json change only for tiles you drew
  real damage on.
* Resumable: already-resolved tiles (and, for negatives, ones you already scrolled past)
  are skipped next time. Use --all to force them back.  --dry-run reviews without writing.
"""
import os
import re
import sys
import json
import time
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path

from PIL import Image

DATA = Path(os.environ.get("SEED30CM_DIR", "data/seed30cm"))
IMG = DATA / "images"
LAB = DATA / "labelme"          # canonical JSONs that labelme_to_masks.py reads
MASK = DATA / "masks"
PRI = DATA / "priors"
WORK = DATA / "review_work"     # scratch folder we hand to Labelme (hardlinks, not copies)
MANIFEST = Path("dataset_manifest.py")
PYEXE = sys.executable          # the ads_env python running this script

# label meanings ------------------------------------------------------------
GUIDE_LABELS = {"prior", "hint", "_prior"}                  # grey guide, ignored everywhere
IGNORE_LABELS = {"ignore", "unsure", "dontcare", "skip"}    # don't-care region within a kept tile
REJECT_LABELS = {"reject", "drop"}                          # drop the whole tile
CLEAN_LABELS = {"clean", "keep"}                            # confirmed clean negative (empty mode)
MARKER_LABELS = REJECT_LABELS | CLEAN_LABELS                # verdict markers, never geometry

# what Labelme offers in the label popup, per mode
LABELS_BY_JOB = {"negatives": "damage,reject", "empty": "damage,reject,clean"}

# the 10 opened-empty tiles (kept for the default `empty` run)
EMPTY_TILES = ["0104", "0108", "0117", "0126", "0134", "0135", "0146", "0155", "0167", "0196"]


def z(n):
    return f"{int(n):04d}"


# ---- prior guide (grey outline of the ADS survey polygon) -------------------
def prior_shapes(iid):
    """Vectorise priors/<id>.png into 'prior' guide polygons (ignored by the mask step)."""
    p = PRI / f"{iid}.png"
    if not p.exists():
        return []
    import numpy as np
    import cv2
    m = (np.array(Image.open(p).convert("L")) > 128).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 80:
            continue
        eps = 0.004 * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append({"label": "prior", "points": a.astype(float).tolist(),
                        "group_id": None, "shape_type": "polygon", "flags": {}})
    return out


# ---- dataset_manifest.py list editing (pure, so it is easy to test) ----------
def get_list(text, name):
    """Return the set of ints in a `NAME = [ ... ]` assignment (empty set if absent)."""
    m = re.search(rf"^\s*{name}\s*=\s*\[([^\]]*)\]", text, re.M)
    return {int(x) for x in re.findall(r"-?\d+", m.group(1))} if m else set()


def set_list(text, name, values):
    """Rewrite the `NAME = [ ... ]` line with sorted `values`, preserving any trailing comment."""
    body = "[" + ", ".join(str(v) for v in sorted(values)) + "]"
    pat = re.compile(rf"^(?P<pre>\s*{name}\s*=\s*)\[[^\]]*\](?P<post>.*)$", re.M)
    if not pat.search(text):
        raise RuntimeError(f"could not find `{name} = [...]` in {MANIFEST}")
    return pat.sub(lambda m: f"{m.group('pre')}{body}{m.group('post')}", text, count=1)


def plan_manifest_text(text, reject_ids, clean_ids, damage_ids):
    """Return (new_text, DELETE, NO_DAMAGE) after applying the verdicts to the two lists."""
    rej, cln, dmg = set(map(int, reject_ids)), set(map(int, clean_ids)), set(map(int, damage_ids))
    D = (get_list(text, "DELETE") | rej) - cln - dmg
    N = (get_list(text, "NO_DAMAGE") | cln) - rej - dmg          # damaged tiles go UNLISTED -> auto-damage
    text = set_list(text, "DELETE", D)
    text = set_list(text, "NO_DAMAGE", N)
    return text, D, N


def edit_manifest(reject_ids, clean_ids, damage_ids):
    text = MANIFEST.read_text(encoding="utf-8")
    touched = set(map(int, reject_ids + clean_ids + damage_ids))
    for name in ("RISKY", "TOO_SMALL", "TREECUT_IGNORE", "DAMAGE_IGNORE"):
        clash = get_list(text, name) & touched
        if clash:
            print(f"  NOTE: {sorted(clash)} also appear in {name} (left untouched — check by hand)")
    new_text, D, N = plan_manifest_text(text, reject_ids, clean_ids, damage_ids)
    if new_text != text:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = MANIFEST.with_suffix(f".py.bak.{stamp}")
        bak.write_text(text, encoding="utf-8")
        MANIFEST.write_text(new_text, encoding="utf-8")
        print(f"  edited {MANIFEST}  (backup -> {bak.name})")
    else:
        print(f"  {MANIFEST} unchanged")
    return D, N


# ---- seen-set (so a negatives sweep resumes where you left off) --------------
def seen_path(job):
    return WORK / f"{job}_seen.json"


def read_seen(job):
    p = seen_path(job)
    return set(json.loads(p.read_text())) if p.exists() else set()


def write_seen(job, ids):
    seen_path(job).parent.mkdir(parents=True, exist_ok=True)
    seen_path(job).write_text(json.dumps(sorted(ids)))


def has_damage_mask(iid):
    p = MASK / f"{iid}.png"
    if not p.exists():
        return False
    import numpy as np
    return bool((np.array(Image.open(p)) > 128).any())


# ---- candidate selection ----------------------------------------------------
def candidates(job, ids_override):
    if job == "empty":
        return list(ids_override) if ids_override else list(EMPTY_TILES)
    import pandas as pd
    d = pd.read_csv(DATA / "index.csv", dtype={"id": str})
    d["id"] = d["id"].str.zfill(4)
    return sorted(d[d.role == "negative"].id.tolist())


def unresolved(job, ids, force_all):
    """Drop tiles that already have a committed outcome (or, for negatives, were already seen)."""
    if force_all:
        return list(ids), []
    text = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
    done = get_list(text, "DELETE")
    if job == "empty":
        done |= get_list(text, "NO_DAMAGE")
    seen = read_seen(job) if job == "negatives" else set()
    todo, skip = [], []
    for i in ids:
        if int(i) in done or i in seen or has_damage_mask(i):
            skip.append(i)
        else:
            todo.append(i)
    return todo, skip


# ---- staging: build a temp folder holding only the tiles under review -------
def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)               # hardlink: instant, no extra disk, same NTFS volume
    except OSError:
        shutil.copy2(src, dst)          # fallback if hardlink is unavailable


def seed_json(tempdir, iid, seed_prior):
    """Write tempdir/<id>.json: reuse the canonical annotation if any, else a blank one."""
    canonical = LAB / f"{iid}.json"
    if canonical.exists():
        data = json.loads(canonical.read_text(encoding="utf-8"))
    else:
        with Image.open(IMG / f"{iid}.png") as im:
            W, H = im.size
        data = {"version": "5.5.0", "flags": {}, "shapes": [],
                "imagePath": "", "imageData": None, "imageHeight": H, "imageWidth": W}
    data["imagePath"] = f"{iid}.png"    # points at the hardlink sitting next to it
    data["imageData"] = None
    if seed_prior:
        labs = {str(s.get("label", "")).lower() for s in data.get("shapes", [])}
        if not (labs - GUIDE_LABELS) and not (labs & GUIDE_LABELS):   # nothing real, no guide yet
            data["shapes"] = prior_shapes(iid) + data.get("shapes", [])
    (tempdir / f"{iid}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def stage(job, ids, seed_prior):
    tempdir = WORK / job
    if tempdir.exists():
        shutil.rmtree(tempdir)
    tempdir.mkdir(parents=True)
    for iid in ids:
        link_or_copy(IMG / f"{iid}.png", tempdir / f"{iid}.png")
        seed_json(tempdir, iid, seed_prior)
    return tempdir


# ---- launch + classify ------------------------------------------------------
def launch(tempdir, labels):
    cmd = [PYEXE, "-m", "labelme", str(tempdir), "--output", str(tempdir),
           "--labels", labels, "--no-sort-labels"]
    print("\nLaunching Labelme — review your tiles, then CLOSE the window when done.")
    print("   " + " ".join(cmd) + "\n")
    subprocess.run(cmd)


def classify(job, jpath):
    """One tile -> verdict ('damage'|'reject'|'clean'|'untouched'), plus has-real-damage flag."""
    data = json.loads(Path(jpath).read_text(encoding="utf-8"))
    labels = [str(s.get("label", "")).lower() for s in data.get("shapes", []) if s.get("points")]
    reject = any(l in REJECT_LABELS for l in labels)
    clean = any(l in CLEAN_LABELS for l in labels)
    damage = any(l not in (GUIDE_LABELS | IGNORE_LABELS | MARKER_LABELS) for l in labels)
    if reject:                          # an explicit "drop this" always wins
        verdict = "reject"
    elif damage:
        verdict = "damage"
    elif job == "empty" and clean:
        verdict = "clean"
    else:
        verdict = "untouched"
    return verdict, damage


def reconcile(job, tempdir, ids, stage_time):
    plan = {"damage": [], "reject": [], "clean": [], "untouched": [], "seen": [], "_real": {}}
    for iid in ids:
        j = tempdir / f"{iid}.json"
        if not j.exists():
            plan["untouched"].append(iid)
            continue
        if j.stat().st_mtime > stage_time + 0.5:     # Labelme re-saved it -> you looked at it
            plan["seen"].append(iid)
        verdict, has_damage = classify(job, j)
        plan[verdict].append(iid)
        plan["_real"][iid] = has_damage
    return plan


# ---- apply ------------------------------------------------------------------
def copy_back_annotations(tempdir, ids, real_shapes):
    """Persist annotated tiles into canonical labelme/ (marker shapes stripped) for the mask step."""
    LAB.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        if not real_shapes.get(iid):
            continue
        data = json.loads((tempdir / f"{iid}.json").read_text(encoding="utf-8"))
        data["shapes"] = [s for s in data.get("shapes", [])
                          if str(s.get("label", "")).lower() not in MARKER_LABELS]
        data["imagePath"] = os.path.relpath(IMG / f"{iid}.png", LAB).replace(os.sep, "/")
        data["imageData"] = None
        (LAB / f"{iid}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def run(script):
    print(f"\n$ python {script}")
    subprocess.run([PYEXE, script])


def apply_plan(tempdir, plan):
    copy_back_annotations(tempdir, plan["damage"], plan["_real"])
    edit_manifest(plan["reject"], plan["clean"], plan["damage"])
    run("labelme_to_masks.py")          # rasterise annotations -> masks
    run("dataset_manifest.py")          # rebuild manifest.csv from the buckets


# ---- summary + CLI ----------------------------------------------------------
def print_plan(job, plan, skipped):
    def show(tag, key, note):
        ids = plan[key]
        if ids:
            print(f"  {tag:22s} ({len(ids):>3}): {', '.join(ids)}")
            print(f"  {'':22s}       {note}")
    print("\n" + "=" * 72)
    print(f"REVIEW SUMMARY — {job}")
    print("=" * 72)
    show("DAMAGE -> positive", "damage", "become POSITIVES (masks rebuilt)")
    show("REJECT -> DELETE", "reject", "dropped from training")
    if job == "empty":
        show("CLEAN  -> NO_DAMAGE", "clean", "become confirmed NEGATIVES")
    show("kept / untouched", "untouched", "left as-is (negatives stay negative)")
    if skipped:
        print(f"  already resolved       ({len(skipped):>3}): skipped (use --all to include)")


LEGENDS = {
    "empty": ("Reviewing the opened-empty tiles. When you DRAW a shape, pick a label:\n"
              "   * has damage  -> polygon(s) labelled \"damage\"  (becomes a POSITIVE)\n"
              "   * bad/unusable-> a quick box labelled \"reject\"  (dropped)\n"
              "   * truly clean -> a quick box labelled \"clean\"   (becomes a NEGATIVE)\n"
              "   * not sure    -> draw nothing; it stays unresolved for next time\n"
              "   The grey outline is the ADS survey's claimed damage — a hint only."),
    "negatives": ("Sweeping the confirmed-negative tiles. When you DRAW a shape, pick a label:\n"
                  "   * looks clean -> draw NOTHING, go to the next tile   (stays a NEGATIVE)\n"
                  "   * has damage  -> polygon(s) labelled \"damage\"        (becomes a POSITIVE)\n"
                  "   * bad tile / won't trace -> a box labelled \"reject\"  (dropped)\n"
                  "   Prefer no drawing? Note the ids and run:  python review_tiles.py drop <ids...>"),
}


def cmd_drop(raw_ids):
    ids = [z(i) for i in raw_ids]
    print(f"Dropping {len(ids)} tile(s) -> DELETE: {', '.join(ids)}")
    edit_manifest(reject_ids=[int(i) for i in ids], clean_ids=[], damage_ids=[])
    run("dataset_manifest.py")
    print("\nDone — those tiles are excluded from training.")


def cmd_review(job, raw_ids, args):
    ids = [z(i) for i in candidates(job, raw_ids)]
    todo, skipped = unresolved(job, ids, args.all)
    if args.start:                                  # resume point, e.g. --from 0492
        todo = [i for i in todo if int(i) >= int(args.start)]
    if args.limit:                                  # review at most N this session
        todo = todo[:args.limit]
    if not todo:
        print(f"Nothing to review for '{job}' — all {len(ids)} tiles are already resolved.")
        print("(Use --all to re-open them anyway.)")
        return
    print(f"\n{LEGENDS[job]}\n\nStaging {len(todo)} tile(s)"
          + (f" ({len(skipped)} already resolved, skipped)" if skipped else "") + " …")

    tempdir = stage(job, todo, seed_prior=(job == "empty"))
    stage_time = time.time()
    launch(tempdir, LABELS_BY_JOB[job])
    plan = reconcile(job, tempdir, todo, stage_time)
    print_plan(job, plan, skipped)

    if job == "negatives":                          # remember what you looked at, for resume
        write_seen(job, read_seen(job) | set(plan["seen"]))

    if not (plan["damage"] or plan["reject"] or plan["clean"]):
        print("\nNo verdicts to apply (nothing drawn). Done.")
        return
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not args.yes:
        ans = input("\nApply these verdicts (edit dataset_manifest.py + rebuild masks)? [Y/n] ").strip().lower()
        if ans in ("n", "no"):
            print("Left everything unchanged. The ids above are yours to apply by hand.")
            return
    apply_plan(tempdir, plan)
    print("\nDone — training set updated.")


def main():
    ap = argparse.ArgumentParser(description="Interactive Labelme review, driven by visible labels.")
    ap.add_argument("job", choices=["negatives", "empty", "drop"], help="what to do")
    ap.add_argument("ids", nargs="*", help="tile ids (required for 'drop'; optional for 'empty')")
    ap.add_argument("--from", dest="start", metavar="ID", help="resume: only review ids >= this (e.g. 0492)")
    ap.add_argument("--limit", type=int, metavar="N", help="review at most N tiles this session")
    ap.add_argument("--all", action="store_true", help="re-stage every tile, even resolved ones")
    ap.add_argument("--yes", action="store_true", help="apply without the confirm prompt")
    ap.add_argument("--dry-run", action="store_true", help="review then print verdicts; change nothing")
    args = ap.parse_args()

    if not IMG.exists():
        sys.exit(f"No images at {IMG}. Run the tile builder first.")
    if args.job == "drop":
        if not args.ids:
            sys.exit("Usage: python review_tiles.py drop <id> [<id> ...]")
        cmd_drop(args.ids)
        return
    cmd_review(args.job, args.ids, args)


if __name__ == "__main__":
    main()
