"""SUPERSEDED — proposal-era demo of the weak-augmentation engine.

Shows that a small set of manually aligned seed samples can generate unlimited
training pairs for an alignment CNN. The alignment approach is retired (see
coarse_align.py); kept because it contains the OSIP tile fetch that data_prep/ reuses.

Based on:
  - Jiang et al. (KDD 2021) — "Weakly Supervised Spatial Deep Learning
    Based on Imperfect Vector Labels with Registration Errors"
  - He et al. (KDD 2022) — "Quantifying and Reducing Registration
    Uncertainty of Spatial Vector Labels on Earth Imagery"


Usage:
    conda activate ads_env
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    python demo_weak_augmentation.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import requests
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# Local imports
from augmentation import SyntheticDisplacer

# ================================================================
# CONFIGURATION
# ================================================================
PKL_PATH = Path("data/paired_samples_2024.pkl")
OUTPUT_DIR = Path("data/weak_aug_demo")

# Oregon 2024 OSIP server URL (for visualization tiles)
NAIP_URL = "https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2024/OSIP_2024_WM/ImageServer"

# Visualization tile: MUCH wider than training tile to show full displacement context
# Must be large enough to contain polygon + displacement + surrounding context
VIS_TILE_EXTENT_M = 1400.0   # 1400m × 1400m (~0.87 miles) — fits 0.4mi displacement
VIS_TILE_PX = 2048           # High-res for crisp visualization

# Curriculum phases — displacement in METERS (geographic, not pixels)
# All 6 affine parameters: tx, ty, rotation, scale_x, scale_y, shear
# Real ADS misalignment: up to 0.4 miles (644m)
PHASES = [
    {
        "name": "Phase 1: Coarse (~0.5 mi)",
        "short": "coarse",
        "max_trans_m": 805.0,     # up to 0.5 miles (805m)
        "max_rot": 45.0,          # up to ±45° rotation
        "scale_x": (0.80, 1.20),  # ±20% anisotropic X scale
        "scale_y": (0.80, 1.20),  # ±20% anisotropic Y scale
        "max_shear": 15.0,        # up to ±15° shear/skew
    },
    {
        "name": "Phase 2: Medium (~0.4 mi)",
        "short": "medium",
        "max_trans_m": 644.0,     # ~0.4 miles (644m)
        "max_rot": 30.0,          # ±30° rotation
        "scale_x": (0.90, 1.10),  # ±10% X scale
        "scale_y": (0.90, 1.10),  # ±10% Y scale
        "max_shear": 8.0,         # ±8° shear
    },
    {
        "name": "Phase 3: Fine (~0.17 mi)",
        "short": "fine",
        "max_trans_m": 273.0,     # ~0.17 miles (273m)
        "max_rot": 15.0,           # ±5° rotation
        "scale_x": (0.95, 1.05),  # ±5% X scale
        "scale_y": (0.95, 1.05),  # ±5% Y scale
        "max_shear": 3.0,         # ±3° shear
    },
]

# Colors that pop against forest green background
# Matches mentor reference image style (orange + white)
COLOR_ALIGNED = "#FFFFFF"      # White — ground truth (like mentor images)
COLOR_DISPLACED = "#FF8C00"    # Dark orange — displaced (like mentor images)
OUTLINE_WIDTH = 3.5            # Thick, prominent outlines

# ================================================================
# NAIP TILE FETCHER (for visualization only)
# ================================================================

def fetch_vis_tile(bbox_proj, tile_size_px=2048):
    """Fetch a LARGE NAIP tile for visualization.

    Uses EPSG:6557 for both bbox and image coordinates.
    Downloads RGB PNG for display purposes.

    Args:
        bbox_proj: (xmin, ymin, xmax, ymax) in EPSG:6557 (meters)
        tile_size_px: pixel size of output

    Returns:
        np.ndarray (H, W, 3) uint8 RGB, or None on failure
    """
    xmin, ymin, xmax, ymax = bbox_proj
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "6557",
        "imageSR": "6557",
        "size": f"{tile_size_px},{tile_size_px}",
        "format": "png",
        "pixelType": "U8",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    url = f"{NAIP_URL}/exportImage"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct or resp.content[:1] == b"{":
                print(f"  Server error on vis tile")
                return None
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"  Vis tile attempt {attempt+1}/3: {e}")
            time.sleep(2 ** attempt)
    return None


# ================================================================
# POLYGON DISPLACEMENT IN GEOGRAPHIC SPACE (METERS)
# ================================================================

def displace_polygon_geo(polygon, tx_m, ty_m, rotation_deg=0.0,
                         scale_x=1.0, scale_y=1.0, shear_deg=0.0):
    """Displace a polygon in geographic space using all 6 affine parameters.

    The 6 affine parameters:
      1. tx (translation X) — horizontal shift in meters
      2. ty (translation Y) — vertical shift in meters
      3. rotation — rotation angle in degrees
      4. scale_x — anisotropic scale in X direction
      5. scale_y — anisotropic scale in Y direction
      6. shear — skew/shear angle in degrees

    Args:
        polygon: shapely Polygon in EPSG:6557 (meters)
        tx_m, ty_m: translation (meters)
        rotation_deg: rotation (degrees)
        scale_x, scale_y: anisotropic scale factors
        shear_deg: shear/skew angle (degrees)

    Returns:
        displaced shapely Polygon
    """
    cx, cy = polygon.centroid.x, polygon.centroid.y

    # 1-2. Apply anisotropic scale around centroid (sx ≠ sy → stretching)
    if abs(scale_x - 1.0) > 0.001 or abs(scale_y - 1.0) > 0.001:
        polygon = affinity.scale(polygon, xfact=scale_x, yfact=scale_y, origin=(cx, cy))

    # 3. Apply shear/skew around centroid
    if abs(shear_deg) > 0.01:
        polygon = affinity.skew(polygon, xs=shear_deg, ys=0, origin=(cx, cy))

    # 4. Apply rotation around centroid
    if abs(rotation_deg) > 0.01:
        polygon = affinity.rotate(polygon, rotation_deg, origin=(cx, cy))

    # 5-6. Apply translation
    polygon = affinity.translate(polygon, xoff=tx_m, yoff=ty_m)

    return polygon


def generate_geo_displacement(rng, phase):
    """Generate random displacement using ALL 6 affine parameters.

    Returns: dict with tx, ty, rot, sx, sy, shear, shift_m
    """
    tx = rng.uniform(-phase["max_trans_m"], phase["max_trans_m"])
    ty = rng.uniform(-phase["max_trans_m"], phase["max_trans_m"])
    rot = rng.uniform(-phase["max_rot"], phase["max_rot"])
    sx = rng.uniform(*phase["scale_x"])
    sy = rng.uniform(*phase["scale_y"])
    shear = rng.uniform(-phase["max_shear"], phase["max_shear"])
    shift_m = np.sqrt(tx**2 + ty**2)
    return {"tx": tx, "ty": ty, "rot": rot, "sx": sx, "sy": sy,
            "shear": shear, "shift_m": shift_m}


# ================================================================
# PLOTTING HELPERS
# ================================================================

def plot_polygon_outline(ax, polygon, bbox, img_w, img_h, color, linewidth=3.5, label=None):
    """Plot polygon as outline only on pixel-coordinate axes. No fill."""
    if polygon.is_empty:
        return
    if polygon.geom_type == "MultiPolygon":
        for i, p in enumerate(polygon.geoms):
            plot_polygon_outline(ax, p, bbox, img_w, img_h, color, linewidth,
                                 label=label if i == 0 else None)
        return
    x_geo, y_geo = polygon.exterior.xy
    # Geo → pixel
    px = [(x - bbox[0]) / (bbox[2] - bbox[0]) * img_w for x in x_geo]
    py = [img_h - (y - bbox[1]) / (bbox[3] - bbox[1]) * img_h for y in y_geo]
    ax.plot(px, py, color=color, linewidth=linewidth, label=label, solid_capstyle="round")


def add_scale_bar(ax, bbox, img_w, img_h):
    """Add a scale bar like the mentor reference images."""
    extent_m = bbox[2] - bbox[0]
    # Pick a nice scale bar length
    if extent_m > 1200:
        bar_m = 400
    elif extent_m > 600:
        bar_m = 200
    else:
        bar_m = 100

    bar_miles = bar_m / 1609.34
    bar_px = bar_m / extent_m * img_w

    # Position: bottom-left
    x0 = img_w * 0.05
    y0 = img_h * 0.95
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", linewidth=4, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 8, y0 + 8], color="white", linewidth=2)
    ax.plot([x0 + bar_px, x0 + bar_px], [y0 - 8, y0 + 8], color="white", linewidth=2)
    ax.text(x0 + bar_px / 2, y0 - 15, f"{bar_m}m ({bar_miles:.2f} mi)",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="white", bbox=dict(facecolor="black", alpha=0.6, pad=2))


# ================================================================
# VISUALIZATION FUNCTIONS
# ================================================================

def visualize_full_frame(sample, phase, save_path, idx, rng):
    """Full-frame view: downloads wide NAIP tile, shows both polygons.

    Matches the reference image style:
      - Wide view (~1500m) showing full context
      - White outline = aligned (ground truth)
      - Orange outline = displaced (augmented)
      - Scale bar at bottom
    """
    aligned_poly = sample["polygon"]
    cx, cy = aligned_poly.centroid.x, aligned_poly.centroid.y

    # Generate random displacement in METERS using all 6 affine parameters
    d = generate_geo_displacement(rng, phase)
    displaced_poly = displace_polygon_geo(
        aligned_poly, d["tx"], d["ty"], d["rot"], d["sx"], d["sy"], d["shear"]
    )
    shift_m = d["shift_m"]

    # Compute a bbox that covers BOTH polygons with generous padding
    combined = unary_union([aligned_poly, displaced_poly])
    bounds = combined.bounds  # (minx, miny, maxx, maxy)
    combined_w = bounds[2] - bounds[0]
    combined_h = bounds[3] - bounds[1]
    max_dim = max(combined_w, combined_h)

    # Make tile at least VIS_TILE_EXTENT_M, or larger if needed
    tile_extent = max(VIS_TILE_EXTENT_M, max_dim + 400)
    tile_buffer = tile_extent / 2.0

    # Center on midpoint between the two polygon centroids
    mid_cx = (aligned_poly.centroid.x + displaced_poly.centroid.x) / 2.0
    mid_cy = (aligned_poly.centroid.y + displaced_poly.centroid.y) / 2.0

    vis_bbox = (
        mid_cx - tile_buffer,
        mid_cy - tile_buffer,
        mid_cx + tile_buffer,
        mid_cy + tile_buffer,
    )

    # Download wide visualization tile
    print(f"      Downloading {tile_extent:.0f}m vis tile...", end=" ", flush=True)
    t0 = time.time()
    naip_vis = fetch_vis_tile(vis_bbox, VIS_TILE_PX)
    dt = time.time() - t0
    if naip_vis is None:
        print(f"FAILED")
        return None
    print(f"{dt:.1f}s")

    H, W = naip_vis.shape[:2]

    # ---- PLOT ----
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(naip_vis)

    # Aligned polygon — WHITE outline (like mentor images)
    plot_polygon_outline(ax, aligned_poly, vis_bbox, W, H,
                         COLOR_ALIGNED, OUTLINE_WIDTH, label="Aligned (ground truth)")

    # Displaced polygon — ORANGE outline (like mentor images)
    plot_polygon_outline(ax, displaced_poly, vis_bbox, W, H,
                         COLOR_DISPLACED, OUTLINE_WIDTH, label="Displaced (augmented)")

    # Arrow from aligned → displaced centroid
    ac = aligned_poly.centroid
    dc = displaced_poly.centroid
    ac_px = (ac.x - vis_bbox[0]) / (vis_bbox[2] - vis_bbox[0]) * W
    ac_py = H - (ac.y - vis_bbox[1]) / (vis_bbox[3] - vis_bbox[1]) * H
    dc_px = (dc.x - vis_bbox[0]) / (vis_bbox[2] - vis_bbox[0]) * W
    dc_py = H - (dc.y - vis_bbox[1]) / (vis_bbox[3] - vis_bbox[1]) * H

    ax.annotate("", xy=(dc_px, dc_py), xytext=(ac_px, ac_py),
                arrowprops=dict(arrowstyle="->, head_width=0.5, head_length=0.4",
                                color="yellow", lw=3))

    # Distance label on arrow
    mid_arrow_x = (ac_px + dc_px) / 2
    mid_arrow_y = (ac_py + dc_py) / 2
    ax.text(mid_arrow_x, mid_arrow_y - 20,
            f"~{shift_m:.0f}m ({shift_m/1609:.2f} mi)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="white", bbox=dict(facecolor="black", alpha=0.7, pad=3))

    # Centroid markers
    ax.plot(ac_px, ac_py, marker="+", color=COLOR_ALIGNED,
            markersize=16, markeredgewidth=3)
    ax.plot(dc_px, dc_py, marker="x", color=COLOR_DISPLACED,
            markersize=16, markeredgewidth=3)

    # Scale bar
    add_scale_bar(ax, vis_bbox, W, H)

    # Legend (semi-transparent box)
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_ALIGNED, linewidth=4, label="Aligned (ground truth)"),
        mlines.Line2D([], [], color=COLOR_DISPLACED, linewidth=4, label="Displaced (augmented)"),
        mlines.Line2D([], [], color="yellow", linewidth=2.5, label=f"Displacement vector"),
    ]
    leg = ax.legend(handles=legend_handles, loc="upper left", fontsize=11,
                    facecolor="black", edgecolor="white", labelcolor="white",
                    framealpha=0.8)

    ax.set_title(
        f"Weak Augmentation — Polygon {idx} | DCA={sample.get('dca_code', '?')} | {phase['name']}",
        fontsize=13, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85)
    )

    # Parameter info box (bottom-left, won't overlap with scale bar)
    info_text = (
        f"shift={shift_m:.0f}m ({shift_m/1609:.2f}mi)\n"
        f"θ={d['rot']:.1f}°  sx={d['sx']:.2f}  sy={d['sy']:.2f}  shear={d['shear']:.1f}°"
    )
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, color="white", fontweight="bold",
            ha="right", va="bottom",
            bbox=dict(facecolor="black", alpha=0.75, pad=4))
    ax.axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="black")
    plt.close()

    return shift_m


def visualize_curriculum_comparison(sample, save_path, idx, rng):
    """Side-by-side comparison of all 3 curriculum phases for ONE sample.

    OPTIMIZATION: Downloads ONE tile centered on the aligned polygon
    and reuses it for all 3 phases (3x faster than downloading per-phase).
    """
    aligned_poly = sample["polygon"]
    cx, cy = aligned_poly.centroid.x, aligned_poly.centroid.y

    # Pre-generate all 3 displacements to find the worst-case bounding box
    displacements = []
    for phase in PHASES:
        d = generate_geo_displacement(rng, phase)
        dp = displace_polygon_geo(
            aligned_poly, d["tx"], d["ty"], d["rot"], d["sx"], d["sy"], d["shear"]
        )
        displacements.append((d, dp))

    # Find bbox that covers ALL displaced polygons across all 3 phases
    all_polys = [aligned_poly] + [dp for _, dp in displacements]
    combined = unary_union(all_polys)
    bounds = combined.bounds
    max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    tile_extent = max(VIS_TILE_EXTENT_M, max_dim + 500)
    tile_buffer = tile_extent / 2.0

    # Center on aligned polygon centroid (stable reference across phases)
    vis_bbox = (cx - tile_buffer, cy - tile_buffer,
                cx + tile_buffer, cy + tile_buffer)

    # Download ONE tile for all 3 panels
    print(f"      Downloading single {tile_extent:.0f}m tile for all 3 phases...", end=" ", flush=True)
    t0 = time.time()
    naip_vis = fetch_vis_tile(vis_bbox, VIS_TILE_PX)
    print(f"{time.time()-t0:.1f}s")

    if naip_vis is None:
        print(f"      FAILED — skipping curriculum for polygon {idx}")
        return

    H, W = naip_vis.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    for col, phase in enumerate(PHASES):
        ax = axes[col]
        d, displaced_poly = displacements[col]
        shift_m = d["shift_m"]

        # Reuse the SAME downloaded tile
        ax.imshow(naip_vis)

        plot_polygon_outline(ax, aligned_poly, vis_bbox, W, H,
                             COLOR_ALIGNED, OUTLINE_WIDTH)
        plot_polygon_outline(ax, displaced_poly, vis_bbox, W, H,
                             COLOR_DISPLACED, OUTLINE_WIDTH)

        # Arrow
        ac, dc = aligned_poly.centroid, displaced_poly.centroid
        ac_px = (ac.x - vis_bbox[0]) / (vis_bbox[2] - vis_bbox[0]) * W
        ac_py = H - (ac.y - vis_bbox[1]) / (vis_bbox[3] - vis_bbox[1]) * H
        dc_px = (dc.x - vis_bbox[0]) / (vis_bbox[2] - vis_bbox[0]) * W
        dc_py = H - (dc.y - vis_bbox[1]) / (vis_bbox[3] - vis_bbox[1]) * H
        ax.annotate("", xy=(dc_px, dc_py), xytext=(ac_px, ac_py),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5))

        add_scale_bar(ax, vis_bbox, W, H)

        ax.set_title(
            f"{phase['name']}",
            fontsize=13, fontweight="bold", color="white",
            pad=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85)
        )
        # Params info box inside plot (bottom-right)
        info = (
            f"shift={shift_m:.0f}m ({shift_m/1609:.2f}mi)\n"
            f"θ={d['rot']:.1f}° sx={d['sx']:.2f} sy={d['sy']:.2f}\n"
            f"shear={d['shear']:.1f}°"
        )
        ax.text(0.98, 0.02, info, transform=ax.transAxes,
                fontsize=9, color="white", fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="black", alpha=0.7, pad=3))
        ax.axis("off")

    # Shared legend
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_ALIGNED, linewidth=4, label="Aligned (ground truth)"),
        mlines.Line2D([], [], color=COLOR_DISPLACED, linewidth=4, label="Displaced (augmented)"),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower right", fontsize=11,
                    facecolor="black", edgecolor="white", labelcolor="white")

    fig.suptitle(
        f"Curriculum Learning — Polygon {idx} | DCA={sample.get('dca_code', '?')}\n"
        f"Training progresses: coarse alignment → medium refinement → fine-grained correction",
        fontsize=15, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.85)
    )
    fig.subplots_adjust(top=0.88)  # Reserve space for suptitle so it doesn't overlap
    plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="#111111")
    plt.close()


def visualize_unlimited_proof(sample, save_path, idx, rng):
    """Same sample, 3 DIFFERENT displacements — proves unlimited data.

    OPTIMIZATION: Downloads ONE wide tile and reuses it for all 4 panels
    (4x faster than downloading per-panel).
    """
    aligned_poly = sample["polygon"]
    cx, cy = aligned_poly.centroid.x, aligned_poly.centroid.y

    # Pre-generate all 3 displacements to find worst-case bbox
    unlimited_phase = {
        "max_trans_m": 400.0, "max_rot": 20.0,
        "scale_x": (0.85, 1.15), "scale_y": (0.85, 1.15), "max_shear": 10.0
    }
    displacements = []
    for _ in range(3):
        d = generate_geo_displacement(rng, unlimited_phase)
        dp = displace_polygon_geo(
            aligned_poly, d["tx"], d["ty"], d["rot"], d["sx"], d["sy"], d["shear"]
        )
        displacements.append((d, dp))

    # Find bbox that covers aligned poly + ALL displaced variants
    all_polys = [aligned_poly] + [dp for _, dp in displacements]
    combined = unary_union(all_polys)
    bounds = combined.bounds
    max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    tile_extent = max(VIS_TILE_EXTENT_M, max_dim + 500)
    tile_buffer = tile_extent / 2.0
    vis_bbox = (cx - tile_buffer, cy - tile_buffer,
                cx + tile_buffer, cy + tile_buffer)

    # Download ONE tile for all 4 panels
    print(f"      Downloading single {tile_extent:.0f}m tile for all 4 panels...", end=" ", flush=True)
    t0 = time.time()
    naip_vis = fetch_vis_tile(vis_bbox, VIS_TILE_PX)
    print(f"{time.time()-t0:.1f}s")

    if naip_vis is None:
        print(f"      FAILED — skipping unlimited proof for polygon {idx}")
        return

    H, W = naip_vis.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))

    # Panel 1: Original aligned polygon only (reuse same tile)
    axes[0].imshow(naip_vis)
    plot_polygon_outline(axes[0], aligned_poly, vis_bbox, W, H,
                         COLOR_ALIGNED, OUTLINE_WIDTH)
    add_scale_bar(axes[0], vis_bbox, W, H)
    axes[0].set_title(f"Seed Sample {idx}\n(always same aligned polygon)",
                      fontsize=12, fontweight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85))
    axes[0].axis("off")

    # Panels 2-4: THREE different displacements (reuse same tile)
    for i in range(3):
        ax = axes[i + 1]
        d, displaced_poly = displacements[i]
        shift_m = d["shift_m"]

        ax.imshow(naip_vis)
        plot_polygon_outline(ax, aligned_poly, vis_bbox, W, H,
                             COLOR_ALIGNED, OUTLINE_WIDTH)
        plot_polygon_outline(ax, displaced_poly, vis_bbox, W, H,
                             COLOR_DISPLACED, OUTLINE_WIDTH)

        ac, dc = aligned_poly.centroid, displaced_poly.centroid
        ac_px = (ac.x - vis_bbox[0]) / (vis_bbox[2] - vis_bbox[0]) * W
        ac_py = H - (ac.y - vis_bbox[1]) / (vis_bbox[3] - vis_bbox[1]) * H
        dc_px = (dc.x - vis_bbox[0]) / (vis_bbox[2] - vis_bbox[0]) * W
        dc_py = H - (dc.y - vis_bbox[1]) / (vis_bbox[3] - vis_bbox[1]) * H
        ax.annotate("", xy=(dc_px, dc_py), xytext=(ac_px, ac_py),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5))
        add_scale_bar(ax, vis_bbox, W, H)

        ax.set_title(
            f"Displacement #{i+1}",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85)
        )
        info = (
            f"shift={shift_m:.0f}m ({shift_m/1609:.2f}mi)\n"
            f"θ={d['rot']:.1f}° sx={d['sx']:.2f} sy={d['sy']:.2f}\n"
            f"shear={d['shear']:.1f}°"
        )
        ax.text(0.98, 0.02, info, transform=ax.transAxes,
                fontsize=9, color="white", fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(facecolor="black", alpha=0.7, pad=3))
        ax.axis("off")

    fig.suptitle(
        f"PROOF: Same seed sample → 3 DIFFERENT training pairs (unlimited data from 22 samples)",
        fontsize=15, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.85)
    )
    fig.subplots_adjust(top=0.88)
    plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="#111111")
    plt.close()


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("WEAK AUGMENTATION ENGINE — PROOF OF CONCEPT")
    print("=" * 70)
    print(f"\nBased on:")
    print(f"  • Jiang et al. (KDD 2021) — Weakly Supervised Spatial DL")
    print(f"  • He et al. (KDD 2022) — Registration Uncertainty Reduction")

    # Load data
    print(f"\nLoading {PKL_PATH}...")
    with open(PKL_PATH, "rb") as f:
        samples = pickle.load(f)
    print(f"  Loaded {len(samples)} paired samples")
    print(f"  Visualization tile extent: {VIS_TILE_EXTENT_M:.0f}m ({VIS_TILE_EXTENT_M/1609:.2f} miles)")
    print(f"  Max displacement (Phase 1): {PHASES[0]['max_trans_m']:.0f}m ({PHASES[0]['max_trans_m']/1609:.2f} miles)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)  # Reproducible but random

    # Select diverse samples
    test_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    test_indices = [i for i in test_indices if i < len(samples)]

    total_start = time.time()

    # ================================================================
    # DEMO 1: Full-frame views for each curriculum phase
    # ================================================================
    print(f"\n{'='*70}")
    print(f"DEMO 1: Full-Frame Augmentation Views (wide NAIP tiles)")
    print(f"{'='*70}")
    for idx in test_indices:
        sample = samples[idx]
        print(f"\n  Polygon {idx}: DCA={sample.get('dca_code', '?')}, "
              f"original shift={sample.get('shift_meters', 0):.0f}m")

        for phase in PHASES:
            save_path = OUTPUT_DIR / f"fullframe_{idx:02d}_{phase['short']}.png"
            shift = visualize_full_frame(sample, phase, save_path, idx, rng)
            if shift:
                print(f"    → {phase['short']}: shift={shift:.0f}m → {save_path.name}")

    # ================================================================
    # DEMO 2: Curriculum comparison (3 phases side-by-side)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"DEMO 2: Curriculum Comparison (coarse → medium → fine)")
    print(f"{'='*70}")
    for idx in test_indices:
        sample = samples[idx]
        print(f"\n  Polygon {idx}:")
        save_path = OUTPUT_DIR / f"curriculum_{idx:02d}.png"
        visualize_curriculum_comparison(sample, save_path, idx, rng)
        print(f"    → {save_path.name}")

    # ================================================================
    # DEMO 3: Same sample → 3 different displacements (unlimited data)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"DEMO 3: Same Sample → Different Displacements (proves unlimited data)")
    print(f"{'='*70}")
    for idx in test_indices:
        sample = samples[idx]
        print(f"\n  Polygon {idx}:")
        save_path = OUTPUT_DIR / f"proof_unlimited_{idx:02d}.png"
        visualize_unlimited_proof(sample, save_path, idx, rng)
        print(f"    → {save_path.name}")

    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - total_start
    total_files = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n{'='*70}")
    print(f"DONE! ({total_time:.0f}s)")
    print(f"{'='*70}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Total visualizations: {total_files}")
    print(f"\n  Style: white=aligned, orange=displaced (matching mentor reference images)")
    print(f"  Each view: ~{VIS_TILE_EXTENT_M:.0f}m tile with scale bar")
    print(f"\n  For GSoC proposal, include the 'curriculum_*.png' and 'proof_unlimited_*.png' images.")


if __name__ == "__main__":
    main()
