"""
annotate_regions.py  — LOCAL interactive tool (run on your PC, NOT Colab)
=========================================================================
Redraw the tight damage boundary on each 30cm tile produced by
build_30cm_seed_tiles.py. Saves a binary ground-truth mask per tile, pixel-aligned
to the image, so it can train the segmenter directly.

The rough (misaligned) ADS polygon is shown in ORANGE as a hint. You draw the CORRECT
tight boundary around the damage REGIONS (clusters of dead/brown crowns) — NOT every
individual tree. The title shows the survey's damage cause + severity.

This uses a simple click-based drawer (no matplotlib PolygonSelector, which was crashing)
and it CLEARS matplotlib's default keyboard shortcuts so each key does exactly one thing.

Controls (click the image window first so it has focus):
  LEFT click        drop a polygon vertex
  RIGHT click       close/finish the current polygon (or click near the first dot)
  a                 finish current polygon and start ANOTHER on the same tile
  z                 undo the last vertex of the current polygon
  d                 delete the last FINISHED polygon
  r                 reset the current (unfinished) polygon
  enter / n         save this tile's mask (all polygons) and go to next
  s                 skip = save an EMPTY mask (no clear damage here)
  q                 quit (progress kept; rerun to resume)

Tips: tiles are large (~1900px) — MAXIMIZE the window. Draw a loose blob around each
dead-tree CLUSTER; if damage is scattered/unclear, press `s` to skip (an empty mask is
a useful "no-damage" example, not a waste).

Run:
  conda activate ads_env
  python annotate_regions.py
"""
from pathlib import Path

import numpy as np

import matplotlib
# Pick an interactive backend that works on Windows: try Tk, then Qt.
for _bk in ("TkAgg", "Qt5Agg"):
    try:
        matplotlib.use(_bk)
        break
    except Exception:
        continue
# Disable matplotlib's built-in key shortcuts (s=save, q=quit, a/r/etc.) so they can't
# steal our keys or pop dialogs. We handle all keys ourselves.
for _k in list(matplotlib.rcParams):
    if _k.startswith("keymap."):
        matplotlib.rcParams[_k] = []
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image, ImageDraw

try:
    import pandas as pd
except Exception:
    pd = None

# ------------------------------------------------------------------ config
DATA = Path("data/seed30cm")
IMG_DIR = DATA / "images"
PRIOR_DIR = DATA / "priors"
MASK_DIR = DATA / "masks"
INDEX_CSV = DATA / "index.csv"
MASK_DIR.mkdir(parents=True, exist_ok=True)
FORCE_REDO = False          # True = re-annotate tiles that already have a mask


# ---------------------------------------------------------------- pure logic
# These functions mutate a plain-dict `state` and are unit-tested headlessly, so the
# interactive behaviour is verified without a live window.
def make_state():
    return {"polys": [], "cur": [], "action": None}


def _commit(state):
    """Finish the current polygon (if it has >=3 points) and start a fresh one."""
    if len(state["cur"]) >= 3:
        state["polys"].append(list(state["cur"]))
    state["cur"] = []


def handle_click(state, x, y, button, close_thresh):
    """LEFT (button 1) = add a vertex, or close if clicked near the first vertex.
    RIGHT (button 3) = close the current polygon."""
    if x is None or y is None:          # click outside the image
        return
    if button == 3:
        _commit(state)
        return
    if button == 1:
        cur = state["cur"]
        if len(cur) >= 3 and (abs(x - cur[0][0]) <= close_thresh
                              and abs(y - cur[0][1]) <= close_thresh):
            _commit(state)
        else:
            cur.append((x, y))


def handle_key(state, key):
    """Return True if the tile is finished (save/skip/quit), else False."""
    if key in ("enter", "n"):
        _commit(state); state["action"] = "save"
    elif key == "a":
        _commit(state)
    elif key == "r":
        state["cur"] = []
    elif key == "z":
        if state["cur"]:
            state["cur"].pop()
    elif key == "d":
        if state["polys"]:
            state["polys"].pop()
    elif key == "s":
        state["action"] = "skip"
    elif key == "q":
        state["action"] = "quit"
    return state["action"] is not None


def rasterize_polys(polys, H, W):
    """Fill each closed polygon (list of (x, y) vertices) into a 0/255 uint8 mask."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for verts in polys:
        if len(verts) >= 3:
            d.polygon([(float(x), float(y)) for x, y in verts], fill=255)
    return np.array(im, np.uint8)


def load_meta():
    """id -> {damage_type, pct_affected, dca, host, ...} from index.csv (optional)."""
    meta = {}
    if pd is not None and INDEX_CSV.exists():
        try:
            df = pd.read_csv(INDEX_CSV, dtype=str)
            meta = {str(r["id"]): r.to_dict() for _, r in df.iterrows()}
        except Exception as e:
            print(f"  (could not read {INDEX_CSV}: {e})")
    return meta


# ---------------------------------------------------------------- interactive
def annotate_tile(img_path: Path, meta=None, progress=""):
    """Show one tile; return (action, mask). action in {'save','skip','quit'}."""
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    close_thresh = 0.02 * max(H, W)
    prior_path = PRIOR_DIR / img_path.name
    prior = (np.array(Image.open(prior_path).convert("L")) > 128
             if prior_path.exists() else None)

    state = make_state()

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(img)
    if prior is not None and prior.any():
        ax.contour(prior, levels=[0.5], colors=["orange"], linewidths=1.2)

    m = (meta or {}).get(img_path.stem, {})
    info = ""
    if m:
        host = m.get("host", "")
        host = "" if host in (None, "", "nan", "NaN") else f" on {host}"
        info = f"   {m.get('damage_type','')} · {m.get('pct_affected','')} · {m.get('dca','')}{host}"
    ax.set_title(f"{progress} {img_path.name}{info}\n"
                 "orange = rough ADS hint  |  LEFT-click vertices, RIGHT-click to close\n"
                 "[a] next poly  [z] undo pt  [d] del poly  [r] reset  "
                 "[enter] save+next  [s] skip  [q] quit",
                 fontsize=9)
    ax.axis("off")

    cur_line, = ax.plot([], [], "-o", color="red", lw=1.6, ms=5, mfc="yellow")
    patches = []

    def render():
        for p in patches:
            p.remove()
        patches.clear()
        for poly in state["polys"]:
            patch = MplPolygon(poly, closed=True, facecolor="red", alpha=0.30,
                               edgecolor="yellow", lw=1.6)
            ax.add_patch(patch)
            patches.append(patch)
        if state["cur"]:
            cur_line.set_data([v[0] for v in state["cur"]],
                              [v[1] for v in state["cur"]])
        else:
            cur_line.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        handle_click(state, event.xdata, event.ydata, event.button, close_thresh)
        render()

    def on_key(event):
        finished = handle_key(state, event.key)
        if finished:
            plt.close(fig)
        else:
            render()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)

    mask = np.zeros((H, W), np.uint8)
    if state["action"] == "save" and state["polys"]:
        mask = rasterize_polys(state["polys"], H, W)
    return state["action"], mask


def main():
    tiles = sorted(IMG_DIR.glob("*.png"))
    if not tiles:
        print(f"No tiles in {IMG_DIR}. Run build_30cm_seed_tiles.py first.")
        return

    meta = load_meta()
    todo = [t for t in tiles if FORCE_REDO or not (MASK_DIR / t.name).exists()]
    print(f"{len(tiles)} tiles total | {len(tiles) - len(todo)} already labeled | "
          f"{len(todo)} to do this run.")
    if not todo:
        print("Nothing to label. Set FORCE_REDO=True to redo existing masks.")
        return

    done = 0
    for k, img_path in enumerate(todo, 1):
        out = MASK_DIR / img_path.name
        action, mask = annotate_tile(img_path, meta, progress=f"[{k}/{len(todo)}]")
        if action == "quit":
            print(f"\nStopped. {done} labeled this run. Rerun to resume.")
            return
        Image.fromarray(mask).save(out)
        done += 1
        n_px = int((mask > 0).sum())
        pct = 100.0 * n_px / mask.size
        print(f"  {img_path.name}: {'saved' if action == 'save' else 'skipped'} "
              f"({n_px} damage px, {pct:.1f}% of tile)")
    print(f"\nAll tiles labeled. {done} this run. Masks in {MASK_DIR}.")


if __name__ == "__main__":
    main()
