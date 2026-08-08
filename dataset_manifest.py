"""
dataset_manifest.py — decide WHICH tiles train the model, from an explicit curation.
=====================================================================================
You hand-sorted the tiles into buckets; this turns that into the authoritative include
list. Edit the BUCKETS below (plain ints, auto zero-padded), then run:

    python dataset_manifest.py

It writes `data/seed30cm/manifest.csv` (the list `finetune_30cm.py` reads) and prints how
many tiles will train the model, split into damage vs negative. It is NON-DESTRUCTIVE:
it only READS the masks and WRITES manifest.csv — it never edits your labels or masks.

Rules (so you can predict what each bucket does):
* DELETE / RISKY / TOO_SMALL      -> excluded from training.
* NO_DAMAGE                        -> a clean negative (empty) example. If the tile was
                                      never saved in Labelme (no mask), it's still included
                                      as an empty negative on the fly — no re-opening needed.
                                      If its mask actually HAS damage, it's used as DAMAGE and
                                      you get a warning (fix the bucket if that's wrong).
* TREECUT_IGNORE                   -> if the tile has damage: used as DAMAGE (+ its ignore
                                      region is excluded from the loss). If it's pure cut
                                      forest (no damage): excluded — same effect as ignoring
                                      the whole tile, and it avoids teaching bare/log ground
                                      as "healthy forest".
* DAMAGE_IGNORE                    -> damage tile that also has a big ignore area.
* anything you did NOT list        -> DAMAGE if its mask has damage, else EXCLUDED (an
                                      un-categorized empty tile is not trusted as a negative).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

DATA = Path("data/seed30cm")
IMG, MASK, IGN, LAB = DATA / "images", DATA / "masks", DATA / "ignore", DATA / "labelme"
DRAFT_VERSION = "5.5.0"                      # prelabel stamp = un-reviewed draft

# ---- YOUR CURATION — edit these lists ---------------------------------------
DELETE         = [0, 439, 451, 458, 461, 468, 469, 471, 472, 477, 478, 481, 483, 486, 487, 490]                                             # drop entirely (bad tile)
RISKY          = [2, 9, 12, 33, 49, 50, 104, 108, 117, 126, 134, 135, 146, 155, 167, 196,  # uncertain -> excluded
                  # 2026-07-18 beetle batch — skip/risky, ignore entirely:
                  92, 114, 138, 142, 176, 177, 182, 183, 187, 194, 222, 230, 241, 242, 243,
                  244, 246, 247, 264, 271, 273, 285, 288, 289, 291, 295, 298, 299, 301, 329,
                  331, 336, 338, 363]
TOO_SMALL      = [31]                                            # tiny unlabeled damage -> exclude
NO_DAMAGE      = [11, 21, 26, 35, 37, 39, 40, 44, 45, 46, 57,    # confirmed clean negatives
                  # 2026-07-18 beetle batch — ADS fired but no real damage (valuable HARD negatives):
                  88, 90, 102, 110, 130, 144, 180, 212, 223, 231, 236, 258, 305, 314, 335, 351, 357]
TREECUT_IGNORE = [16, 24, 25, 27, 30, 32, 38, 60, 68, 69, 72, 78, 79, 81, 84]  # cut forest
DAMAGE_IGNORE  = [34]                                            # damage + a big ignore area
ON_HOLD        = [266, 408, 419]                                 # 2026-07-18: review later -> excluded for now
# (Move an id between lists and re-run to change the training set — that's the "dynamic" part.)

# Cap how many confirmed negatives actually TRAIN. You built 150, but if you have ~95 damage tiles,
# using all 150 makes the set 60%+ empty -> the model gets biased toward predicting "no damage" (low
# recall). A good ratio is ~1:2 negatives:positives. Set to e.g. 50 for now; None = use all 150.
NEG_MAX = None   # e.g. 50  (kept negatives are chosen evenly across the state by spatial cluster)

# Confirmed negatives (ids >=437 from build_diverse_seed_tiles.py) are only TRUSTED once you've
# eyeballed them in Labelme — the first 55 you swept turned out ~38% not-clean (real damage / junk),
# so the un-reviewed ones can't be assumed empty. You reviewed 0437-0491; the rest stay EXCLUDED
# until reviewed. Widen this range (or set to None to trust all) as you review more.
NEG_REVIEWED = set(range(437, 492))   # 0437-0491 = the 55 you reviewed; grow this as you review more


def z(n):
    return f"{int(n):04d}"


BUCKET = {}
for name, lst in [("delete", DELETE), ("risky", RISKY), ("too_small", TOO_SMALL),
                  ("no_damage", NO_DAMAGE), ("treecut_ignore", TREECUT_IGNORE),
                  ("damage_ignore", DAMAGE_IGNORE), ("on_hold", ON_HOLD)]:
    for n in lst:
        if z(n) in BUCKET:
            print(f"  WARN {z(n)} is in two buckets ({BUCKET[z(n)]} & {name}) — using {name}")
        BUCKET[z(n)] = name


def reviewed(i):
    j = LAB / f"{i}.json"
    return j.exists() and str(json.loads(j.read_text(encoding="utf-8")).get("version", "")) != DRAFT_VERSION


def dmg_px(i):
    p = MASK / f"{i}.png"
    return int((np.array(Image.open(p)) > 128).sum()) if p.exists() else -1   # -1 = no mask


def has_ignore(i):
    p = IGN / f"{i}.png"
    return bool(p.exists() and (np.array(Image.open(p)) > 128).any())


def main():
    ids = sorted(p.stem for p in IMG.glob("*.png"))
    # build_diverse_seed_tiles.py writes confirmed negatives with role="negative" in index.csv (they are
    # empty-by-construction: inside SURVEYED_AREAS, far from any damage). Honor that so they are auto-included
    # as negatives WITHOUT the user hand-listing ~150 ids in NO_DAMAGE below.
    idx_role = {}
    idx_csv = DATA / "index.csv"
    if idx_csv.exists():
        _ix = pd.read_csv(idx_csv, dtype={"id": str})
        if "role" in _ix.columns:
            idx_role = {z(r["id"]): str(r["role"]).strip().lower() for _, r in _ix.iterrows()}
    rows, warns = [], []
    for i in ids:
        b = BUCKET.get(i, "(unlisted)")
        d = dmg_px(i); has_mask = d >= 0; hasdmg = d > 0; ign = has_ignore(i)
        if b in ("delete", "risky", "too_small", "on_hold"):
            role, use = "exclude", False
        elif b == "no_damage":
            if hasdmg:
                role, use = "damage", True
                warns.append(f"{i}: bucketed NO-DAMAGE but mask has {d} damage px -> using as DAMAGE "
                             f"(fix the bucket if that's wrong)")
            else:
                role, use = "negative", True
                if not has_mask:
                    warns.append(f"{i}: NO-DAMAGE but never saved in Labelme -> included as empty negative on the fly")
        elif b == "treecut_ignore":
            role, use = ("damage", True) if hasdmg else ("exclude", False)
        elif b == "damage_ignore":
            role, use = "damage", True
            if not hasdmg:
                warns.append(f"{i}: bucketed damage+ignore but mask has 0 damage px")
        else:  # unlisted
            if hasdmg:
                role, use = "damage", True
                if idx_role.get(i) == "negative":
                    warns.append(f"{i}: index.csv role=negative but mask has {d} damage px -> using as DAMAGE")
            elif idx_role.get(i) == "negative":
                if NEG_REVIEWED is None or int(i) in NEG_REVIEWED:
                    role, use = "negative", True                # reviewed confirmed negative -> trusted
                else:
                    role, use = "exclude", False                # un-reviewed confirmed negative -> not trusted yet
            elif has_mask:
                role, use = "exclude", False
                warns.append(f"{i}: empty & UNLISTED -> excluded (add to NO_DAMAGE if it's a clean negative)")
            else:
                role, use = "skip", False                       # un-reviewed draft, not categorized
        rows.append(dict(id=i, bucket=b, has_mask=has_mask, damage=hasdmg, ignore=ign, role=role, use=use))

    df = pd.DataFrame(rows)
    if NEG_MAX is not None:
        neg_idx = df.index[(df.role == "negative") & (df.use)].tolist()
        if len(neg_idx) > NEG_MAX:
            keep = set(np.linspace(0, len(neg_idx) - 1, NEG_MAX).round().astype(int))
            drop = [neg_idx[k] for k in range(len(neg_idx)) if k not in keep]
            df.loc[drop, "use"] = False
            print(f"NEG_MAX={NEG_MAX}: using {NEG_MAX} of {len(neg_idx)} confirmed negatives (rest set use=False)")
    df.to_csv(DATA / "manifest.csv", index=False)
    u = df[df.use]
    nd = int((u.role == "damage").sum()); nn = int((u.role == "negative").sum())
    print(f"\nWrote {DATA / 'manifest.csv'}")
    print(f"TRAINING SET: {nd} damage + {nn} negative = {len(u)} tiles"
          f"  ({int(u.ignore.sum())} carry an ignore region)")
    print(f"  excluded: {int((df.role == 'exclude').sum())}   |  uncategorized drafts skipped: "
          f"{int((df.role == 'skip').sum())}")
    if warns:
        print("\nFLAGS (read these):")
        for w in warns:
            print("  -", w)


if __name__ == "__main__":
    main()
