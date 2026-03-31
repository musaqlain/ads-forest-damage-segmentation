"""
Evaluate Stage 1: ProximityAlign Contour Matching Alignment
============================================================

Tests the CoarseAligner (ProximityAlign-inspired) on seed samples.

Approach (from mentor Paper 3 — "Novel Approaches for Aligning
Geospatial Vector Maps"):
  - Match polygon CONTOUR against damage BOUNDARY edges using a
    distance-transform energy function, NOT NCC on stress area.
  - Three-component energy: mean distance + continuity + variability.
  - Multi-scale search: coarse (step=8) then fine (step=1).

Outputs (in outputs/coarse_eval/):
  - coarse_eval_results.csv     — per-sample metrics table
  - coarse_eval_summary.png     — bar chart of error reduction
  - coarse_eval_grid.png        — 5-column visual grid:
      Col 1: NAIP + original polygon (red) + ground truth (white)
      Col 2: Damage Map (stress + DeepForest gaps)
      Col 3: NAIP + coarse-aligned polygon (cyan) + ground truth
      Col 4: NAIP + ground-truth polygon (white)
      Col 5: Proximity Map (distance to damage edges)

Usage:
    conda activate ads_env
    python evaluate_coarse.py
    python evaluate_coarse.py --search-radius 300
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from shapely.geometry import mapping

# Project imports
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from coarse_align import CoarseAligner
from transforms import make_rasterio_transform, rasterize_polygon, resize_to_square

# ================================================================
# CONFIG
# ================================================================
PKL_PATH   = ROOT / "data" / "paired_samples_2024.pkl"
OUTPUT_DIR = ROOT / "outputs" / "coarse_eval"

# ================================================================
# LOGGING
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
log = logging.getLogger("evaluate_coarse")


# ================================================================
# HELPER: Rasterize a polygon to a binary mask for a given sample
# ================================================================
def rasterize_sample_polygon(sample: dict, polygon, image_size: int) -> np.ndarray:
    """Rasterize a shapely polygon to a binary mask using the sample's bbox.

    Args:
        sample: dict from pkl with 'bbox' key
        polygon: shapely Polygon in project CRS
        image_size: target mask size (square)

    Returns:
        (image_size, image_size) float32 mask
    """
    H = W = image_size
    transform = make_rasterio_transform(sample["bbox"], width=W, height=H)
    return rasterize_polygon(polygon, (H, W), transform)


# ================================================================
# VISUALIZATION HELPERS
# ================================================================
def draw_contour(ax, mask: np.ndarray, color: str, linewidth: float = 2.0,
                 label: str = "") -> None:
    """Draw polygon outline from a binary mask."""
    if mask.max() < 0.01:
        return  # Empty mask, nothing to draw
    try:
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=[linewidth])
    except Exception:
        pass  # Skip if contour fails (e.g., all zeros)
    if label:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def save_alignment_grid(
    samples: list[dict],
    results: list[dict],
    image_size: int,
    save_path: Path,
    max_samples: int = 8,
) -> None:
    """Create a visual grid showing ProximityAlign alignment results.

    5 columns per sample:
      1. NAIP + Original ADS polygon (RED) — the misalignment
      2. Damage Map (stress + DeepForest gap signal)
      3. NAIP + Coarse-aligned polygon (CYAN) — after Stage 1
      4. NAIP + Ground-truth polygon (WHITE) — reference
      5. Proximity Map (distance to damage edges)
    """
    n = min(max_samples, len(results))
    n_cols = 5
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n),
                              squeeze=False)

    fig.suptitle(
        "Stage 1: ProximityAlign Contour Matching — Alignment Results\n"
        "Red = Original (misaligned) | Cyan = Coarse-aligned | White = Ground Truth",
        fontsize=14, fontweight="bold", y=1.01
    )

    col_titles = [
        "Original (misaligned)",
        "Damage Map",
        "Coarse-Aligned Result",
        "Ground Truth Reference",
        "Proximity Map"
    ]

    for row, (sample, res) in enumerate(zip(samples[:n], results[:n])):
        H = W = image_size

        # Prepare NAIP image for display (RGB only, even if 4-band)
        naip_disp_raw = resize_to_square(sample["naip_image"], H)[:, :, :3]
        naip_disp = np.clip(naip_disp_raw.astype(np.float32) / 255.0, 0, 1)

        # Ground-truth aligned mask
        gt_mask = rasterize_sample_polygon(sample, sample["polygon"], H)

        # Original misaligned mask
        orig_mask = res["original_mask"]

        # Coarse-aligned mask
        aligned_mask = res["aligned_mask"]

        # Diagnostic maps from ProximityAlign
        damage_map = res.get("damage_map", np.zeros((H, W)))
        proximity_map = res.get("proximity_map", None)

        # --- Column 1: Original ---
        ax = axes[row, 0]
        ax.imshow(naip_disp, interpolation="bilinear")
        draw_contour(ax, orig_mask, color="red", linewidth=2.5,
                     label="Original ADS")
        draw_contour(ax, gt_mask, color="white", linewidth=1.5,
                     label="Ground Truth")
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
        ax.axis("off")
        dca = sample.get("dca_code", "?")
        shift = sample.get("shift_meters", 0)
        ax.set_title(
            f"Sample {row} | DCA={dca}\nShift: {shift:.0f}m",
            fontsize=8, pad=4
        )
        if row == 0:
            ax.legend(loc="lower left", fontsize=6, framealpha=0.75)

        # --- Column 2: Damage Map ---
        ax = axes[row, 1]
        ax.imshow(naip_disp, interpolation="bilinear", alpha=0.4)
        damage_rgba = plt.cm.hot(damage_map)
        damage_rgba[..., 3] = np.clip(damage_map * 1.5, 0, 0.7)
        ax.imshow(damage_rgba, interpolation="bilinear")
        draw_contour(ax, orig_mask, color="red", linewidth=1.5)
        draw_contour(ax, gt_mask, color="lime", linewidth=1.0)
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
        ax.axis("off")
        ax.set_title("Damage Map\n(bright = damaged)", fontsize=8, pad=4)

        # --- Column 3: Coarse-aligned ---
        ax = axes[row, 2]
        ax.imshow(naip_disp, interpolation="bilinear")
        draw_contour(ax, aligned_mask, color="cyan", linewidth=2.5,
                     label="Coarse-aligned")
        draw_contour(ax, gt_mask, color="white", linewidth=1.5,
                     label="Ground Truth")
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
        ax.axis("off")
        tx, ty = res["tx"], res["ty"]
        score = res["score"]
        ax.set_title(
            f"Stage 1 Result\n"
            f"tx={tx:+.1f}px  ty={ty:+.1f}px  score={score:.2f}",
            fontsize=8, pad=4
        )
        if row == 0:
            ax.legend(loc="lower left", fontsize=6, framealpha=0.75)

        # --- Column 4: Ground truth ---
        ax = axes[row, 3]
        ax.imshow(naip_disp, interpolation="bilinear")
        draw_contour(ax, gt_mask, color="white", linewidth=2.5,
                     label="Ground Truth")
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
        ax.axis("off")
        # Compute centroid error
        err_before = res.get("error_before_px", 0)
        err_after = res.get("error_after_px", 0)
        improvement = err_before - err_after
        ax.set_title(
            f"Ground Truth Ref\n"
            f"Error: {err_before:.0f}px → {err_after:.0f}px "
            f"({'↓' if improvement > 0 else '↑'}{abs(improvement):.0f}px)",
            fontsize=8, pad=4
        )

        # --- Column 5: Proximity Map ---
        ax = axes[row, 4]
        if proximity_map is not None and proximity_map.size > 0:
            ax.imshow(proximity_map, cmap="viridis_r", interpolation="bilinear")
            # Overlay shifted contour in cyan
            draw_contour(ax, aligned_mask, color="cyan", linewidth=1.5)
            draw_contour(ax, gt_mask, color="white", linewidth=1.0)
            ax.set_xlim(0, W - 1)
            ax.set_ylim(H - 1, 0)
            ax.set_title(
                f"Proximity Map\n"
                f"(dark = near damage edge)",
                fontsize=8, pad=4
            )
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title("Proximity Map", fontsize=8, pad=4)
        ax.axis("off")

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].text(0.5, 1.12, title, transform=axes[0, col].transAxes,
                          ha="center", fontsize=9, fontweight="bold",
                          color="#333")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Alignment grid saved → %s", save_path.name)


def save_summary_chart(results: list[dict], save_path: Path) -> None:
    """Bar chart comparing centroid error before/after coarse alignment."""
    n = len(results)
    indices = np.arange(n)
    bar_width = 0.35

    errors_before = [r["error_before_px"] for r in results]
    errors_after  = [r["error_after_px"] for r in results]
    improvements  = [b - a for b, a in zip(errors_before, errors_after)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # Top: Before vs After error bars
    bars1 = ax1.bar(indices - bar_width / 2, errors_before, bar_width,
                     label="Before (original)", color="#EF5350", alpha=0.85)
    bars2 = ax1.bar(indices + bar_width / 2, errors_after, bar_width,
                     label="After (coarse-aligned)", color="#42A5F5", alpha=0.85)

    ax1.set_ylabel("Centroid Error (pixels)", fontsize=11)
    ax1.set_title(
        "Stage 1: ProximityAlign Contour Matching — Centroid Error\n"
        "Measuring pixel distance between predicted and ground-truth polygon centroids",
        fontsize=13, fontweight="bold", pad=12
    )
    ax1.set_xticks(indices)
    ax1.set_xticklabels([f"S{i}" for i in range(n)], fontsize=8)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: Improvement (positive = good)
    colors = ["#4CAF50" if imp > 0 else "#FF7043" for imp in improvements]
    ax2.bar(indices, improvements, color=colors, alpha=0.85)
    ax2.axhline(0, color="#888", linewidth=0.8)
    ax2.set_ylabel("Improvement (pixels)", fontsize=11)
    ax2.set_xlabel("Sample Index", fontsize=11)
    ax2.set_title("Error Reduction per Sample (positive = improved)", fontsize=11)
    ax2.set_xticks(indices)
    ax2.set_xticklabels([f"S{i}" for i in range(n)], fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Stats annotation
    n_improved = sum(1 for imp in improvements if imp > 0)
    mean_imp = np.mean(improvements)
    median_imp = np.median(improvements)
    mean_before = np.mean(errors_before)
    mean_after = np.mean(errors_after)

    stats_text = (
        f"Improved: {n_improved}/{n} samples ({100 * n_improved / n:.0f}%)\n"
        f"Mean error: {mean_before:.1f}px → {mean_after:.1f}px\n"
        f"Mean improvement: {mean_imp:+.1f}px\n"
        f"Median improvement: {median_imp:+.1f}px"
    )
    ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#ccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Summary chart saved → %s", save_path.name)


# ================================================================
# CENTROID UTILITIES
# ================================================================
def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute the centroid (cx, cy) of a binary mask.

    Returns (cx, cy) in pixel coordinates. If the mask is empty,
    returns the center of the image.
    """
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    return float(xs.mean()), float(ys.mean())


def centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Euclidean distance between centroids of two masks (pixels)."""
    cx_a, cy_a = mask_centroid(mask_a)
    cx_b, cy_b = mask_centroid(mask_b)
    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    a = mask_a > 0.5
    b = mask_b > 0.5
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


# ================================================================
# MAIN EVALUATION
# ================================================================
def evaluate(
    search_radius: int = 250,
    image_size: int = 512,
    max_vis_samples: int = 8,
) -> None:
    """Run Stage 1 evaluation on all seed samples."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STAGE 1: PROXIMITYALIGN CONTOUR MATCHING EVALUATION")
    log.info("=" * 60)
    log.info("PKL:            %s", PKL_PATH)
    log.info("Search radius:  %d px", search_radius)
    log.info("Image size:     %d px", image_size)

    # ---- Load data ----
    log.info("\n[1] Loading seed data ...")
    with open(PKL_PATH, "rb") as f:
        samples = pickle.load(f)
    log.info("    %d samples loaded", len(samples))

    # ---- Initialize aligner ----
    aligner = CoarseAligner(
        search_radius_px=search_radius,
    )

    # ---- Run evaluation ----
    log.info("\n[2] Running coarse alignment on REAL misalignment ...")
    results = []
    t0 = time.time()

    for idx, sample in enumerate(samples):
        H = W = image_size

        # Resize NAIP tile — keep ALL bands for aligner (including NIR if present)
        naip_full = resize_to_square(sample["naip_image"], H)
        naip_rgb = naip_full[:, :, :3]  # RGB only for display
        has_nir = naip_full.shape[2] >= 4

        # Ground-truth aligned mask
        gt_mask = rasterize_sample_polygon(sample, sample["polygon"], H)

        # Original misaligned mask
        # Check if we have the original polygon
        if "polygon_original" in sample and sample["polygon_original"] is not None:
            orig_polygon = sample["polygon_original"]
        else:
            log.warning("    Sample %d: no original polygon, skipping", idx)
            continue

        orig_mask = rasterize_sample_polygon(sample, orig_polygon, H)

        # Resize DeepForest mask
        df_mask = sample.get("deepforest_mask", None)
        if df_mask is not None:
            df_resized = resize_to_square(
                (df_mask * 255).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2), H
            )[:, :, 0].astype(np.float32) / 255.0
        else:
            df_resized = None

        # --- Run coarse alignment (with full NAIP including NIR) ---
        tx, ty, score = aligner.align(naip_full, orig_mask, df_resized)

        # Apply the predicted shift to the original mask
        aligned_mask = CoarseAligner.shift_mask(orig_mask, tx, ty)

        # --- Compute metrics ---
        error_before = centroid_distance(orig_mask, gt_mask)
        error_after = centroid_distance(aligned_mask, gt_mask)
        iou_before = mask_iou(orig_mask, gt_mask)
        iou_after = mask_iou(aligned_mask, gt_mask)

        # Get diagnostic maps for visualization
        diagnostics = aligner.get_diagnostic_maps(naip_full, orig_mask, df_resized)

        result = {
            "sample_idx": idx,
            "dca_code": sample.get("dca_code", "?"),
            "shift_meters": sample.get("shift_meters", 0),
            "tx": tx,
            "ty": ty,
            "score": score,
            "error_before_px": error_before,
            "error_after_px": error_after,
            "iou_before": iou_before,
            "iou_after": iou_after,
            "original_mask": orig_mask,
            "aligned_mask": aligned_mask,
            "damage_map": diagnostics["damage_map"],
            "proximity_map": diagnostics.get("proximity_map"),
        }
        results.append(result)

        improvement = error_before - error_after
        status = "✓" if improvement > 0 else "✗"
        log.info(
            "    %s Sample %2d | DCA=%-6s | shift=%5.0fm | "
            "error: %5.1f→%5.1fpx (%+.1f) | IoU: %.2f→%.2f | "
            "conf=%.2f | tx=%+.0f ty=%+.0f",
            status, idx, sample.get("dca_code", "?"),
            sample.get("shift_meters", 0),
            error_before, error_after, -improvement,
            iou_before, iou_after,
            score, tx, ty,
        )

    elapsed = time.time() - t0
    log.info("    Processed %d samples in %.1fs", len(results), elapsed)

    if not results:
        log.error("No results! Check that pkl has 'polygon_original' field.")
        return

    # ---- Summary statistics ----
    log.info("\n[3] Results Summary ...")
    n = len(results)
    n_improved = sum(1 for r in results if r["error_before_px"] > r["error_after_px"])
    mean_before = np.mean([r["error_before_px"] for r in results])
    mean_after = np.mean([r["error_after_px"] for r in results])
    mean_iou_before = np.mean([r["iou_before"] for r in results])
    mean_iou_after = np.mean([r["iou_after"] for r in results])

    log.info("    Samples improved:     %d / %d (%.0f%%)", n_improved, n, 100 * n_improved / n)
    log.info("    Mean centroid error:  %.1f px → %.1f px", mean_before, mean_after)
    log.info("    Mean IoU:             %.3f → %.3f", mean_iou_before, mean_iou_after)
    log.info("    Mean confidence:      %.3f", np.mean([r["score"] for r in results]))

    # ---- Save CSV ----
    csv_path = OUTPUT_DIR / "coarse_eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_idx", "dca_code", "shift_meters",
            "tx", "ty", "score",
            "error_before_px", "error_after_px",
            "iou_before", "iou_after",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: v for k, v in r.items() if k in writer.fieldnames})
    log.info("    CSV saved → %s", csv_path.name)

    # ---- Visualizations ----
    log.info("\n[4] Creating visualizations ...")

    # Alignment grid
    save_alignment_grid(
        samples, results, image_size,
        save_path=OUTPUT_DIR / "coarse_eval_grid.png",
        max_samples=max_vis_samples,
    )

    # Summary bar chart
    save_summary_chart(results, OUTPUT_DIR / "coarse_eval_summary.png")

    # ---- Done ----
    log.info("\n" + "=" * 60)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 60)
    log.info("  Improved: %d/%d samples", n_improved, n)
    log.info("  Error:    %.1f → %.1f px (%.1f%% reduction)",
             mean_before, mean_after,
             100 * (mean_before - mean_after) / max(mean_before, 1e-6))
    log.info("  IoU:      %.3f → %.3f", mean_iou_before, mean_iou_after)
    log.info("  Outputs:  %s", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 ProximityAlign contour matching."
    )
    parser.add_argument("--search-radius", type=int, default=250,
                        help="Search radius in pixels (default: 250)")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Image resize (default: 512)")
    parser.add_argument("--max-vis", type=int, default=8,
                        help="Max samples to visualize in grid (default: 8)")
    args = parser.parse_args()

    evaluate(
        search_radius=args.search_radius,
        image_size=args.image_size,
        max_vis_samples=args.max_vis,
    )
