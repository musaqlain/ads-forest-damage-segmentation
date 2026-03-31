"""
Bridge Script: Convert aligned polygons into training seed data.

This script:
  1. Loads aligned polygons from a GeoPackage.
  2. Uses the OBSERVATION_ID to find each original (misaligned) polygon in the source GDB.
  3. Downloads TWO NAIP tiles per polygon from Oregon 2024 OSIP (high quality, 30cm):
     a. TRAINING TILE: centered on aligned polygon (for weak augmentation)
     b. OVERLAY TILE: covers BOTH original + aligned polygon (for visualization)
  4. Generates DeepForest tree-crown masks (with green-channel fallback).
  5. Saves NAIP tiles as separate PNG files.
  6. Creates overlay visualizations (original vs aligned).
  7. Saves paired_samples as a .pkl file for downstream training/evaluation.

Usage:
    python build_seed_data.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP duplicate lib on Windows
import pickle
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
from collections import Counter
from io import BytesIO
from PIL import Image
from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import unary_union

# ================================================================
# PATHS
# ================================================================
ALIGNED_GPKG = Path("data/aligned_polygons_2024.gpkg")
ORIGINAL_GDB = Path("data/INTERIM_R6_ADS_DATA2024.gdb")
OUTPUT_DIR = Path("data")
OUTPUT_PKL = OUTPUT_DIR / "paired_samples_2024.pkl"
VIS_DIR = OUTPUT_DIR / "overlay_visualizations"
NAIP_DIR = OUTPUT_DIR / "naip_tiles"          # Individual NAIP tile PNGs

# Oregon 2024 OSIP server — high quality, 30cm, guaranteed 2024 imagery
NAIP_URL = "https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2024/OSIP_2024_WM/ImageServer"

# CRS for computation (Oregon Lambert, meters — area-preserving)
PROJECT_CRS = "EPSG:6557"

# Native 30cm resolution — no downsampling
RESOLUTION_M = 0.30

# Training tile: fixed size centered on aligned polygon
# 800m gives ~200m+ buffer on each side beyond a typical polygon,
# providing rich context for the CNN and better proposal visuals.
TRAINING_TILE_EXTENT_M = 800.0   # 800m × 800m → 2667×2667px at 30cm

# Random jitter for tile center (meters) — prevents CNN center bias.
# Without jitter, the polygon is ALWAYS at the image center, and the CNN
# learns the shortcut: "the answer is always the center" instead of actually
# learning to align. With jitter, the polygon appears at random positions.
TILE_CENTER_JITTER_M = 150.0

# Overlay tile: padding around the combined extent of BOTH polygons
OVERLAY_PADDING_M = 100.0        # 100m padding on each side

# Max pixels for 4-band TIFF export — Oregon OSIP server returns 500 errors
# for TIFF requests above ~2700px. We cap at 2048 which is safe.
# The geographic extent stays the same; the server resamples to fit.
# This is acceptable because the CNN resizes to 512px anyway.
TIFF_MAX_PX = 2048

# DeepForest model (loaded once, reused for all tiles)
_deepforest_model = None


def load_aligned_polygons():
    """Load the 22 aligned polygons with their OBSERVATION_ID references."""
    print("=" * 70)
    print("STEP 1: Loading Aligned Polygons")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(ALIGNED_GPKG, layer="aligned_polygons")

    original_crs = gdf.crs
    gdf_proj = gdf.to_crs(PROJECT_CRS)

    print(f"  Source file:  {ALIGNED_GPKG}")
    print(f"  Layer:        aligned_polygons")
    print(f"  Total count:  {len(gdf)}")
    print(f"  Original CRS: {original_crs}")
    print(f"  Project CRS:  {PROJECT_CRS}")
    print()

    # Column info
    print(f"  Columns: {list(gdf.columns)}")
    print()

    # DCA code breakdown
    print(f"  DCA Code Distribution:")
    for code, count in gdf["dca_code"].value_counts().items():
        host = gdf[gdf["dca_code"] == code]["host"].iloc[0] if "host" in gdf.columns else "unknown"
        print(f"    {code}: {count} polygons ({host})")
    print()

    # Geometry statistics (in project CRS = meters)
    areas_m2 = gdf_proj.geometry.area
    print(f"  Polygon Area Statistics (m²):")
    print(f"    Min:    {areas_m2.min():.1f}")
    print(f"    Max:    {areas_m2.max():.1f}")
    print(f"    Mean:   {areas_m2.mean():.1f}")
    print(f"    Median: {areas_m2.median():.1f}")
    print()

    # Check notes column (OBSERVATION_IDs)
    has_notes = gdf["notes"].notna().sum()
    empty_notes = gdf["notes"].isna().sum() + (gdf["notes"] == "").sum()
    print(f"  OBSERVATION_ID links: {has_notes} have IDs, {empty_notes} empty")
    print()

    return gdf, gdf_proj


def load_original_polygons():
    """Load the original DamageCombined polygons, indexed by OBSERVATION_ID."""
    print("=" * 70)
    print("STEP 2: Loading Original (Misaligned) Polygons")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(ORIGINAL_GDB, layer="DamageCombined")

    total = len(gdf)
    mortality = gdf[gdf["DAMAGE_TYPE"] == "Mortality"].copy()
    mortality_proj = mortality.to_crs(PROJECT_CRS)

    print(f"  Source file:  {ORIGINAL_GDB}")
    print(f"  Total polygons: {total}")
    print(f"  Mortality only: {len(mortality)}")
    print()

    # Index by OBSERVATION_ID for fast lookup
    obs_id_map = {}
    for idx, row in mortality_proj.iterrows():
        obs_id = row.get("OBSERVATION_ID")
        if obs_id:
            obs_id_clean = str(obs_id).strip("{}")
            obs_id_map[obs_id_clean] = row

    print(f"  Indexed {len(obs_id_map)} polygons by OBSERVATION_ID")
    print()

    return obs_id_map


def fetch_naip_tile(bbox_proj, tile_size_px, bands=4):
    """Fetch a NAIP tile from the Oregon 2024 OSIP server.

    Uses EPSG:6557 (Oregon Lambert) for BOTH bboxSR and imageSR.
    Attempts to download all 4 bands (R, G, B, NIR) as TIFF.
    Falls back to 3-band PNG if the server doesn't support 4 bands.

    Args:
        bbox_proj: (xmin, ymin, xmax, ymax) in EPSG:6557 (meters)
        tile_size_px: Pixel size of the output tile.
        bands: Number of bands to request (4 = RGBN, 3 = RGB).

    Returns:
        np.ndarray (H, W, bands) uint8, or None on failure.
    """
    xmin, ymin, xmax, ymax = bbox_proj
    bbox_str = f"{xmin},{ymin},{xmax},{ymax}"

    params = {
        "bbox": bbox_str,
        "bboxSR": "6557",
        "imageSR": "6557",
        "size": f"{tile_size_px},{tile_size_px}",
        "pixelType": "U8",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    # Try 4-band TIFF first
    if bands == 4:
        params["format"] = "tiff"
        params["bandIds"] = "0,1,2,3"  # R, G, B, NIR
        # Cap TIFF pixel size — server can't handle large TIFF exports
        if tile_size_px > TIFF_MAX_PX:
            params["size"] = f"{TIFF_MAX_PX},{TIFF_MAX_PX}"
    else:
        params["format"] = "png"

    url = f"{NAIP_URL}/exportImage"

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type or resp.content[:1] == b"{":
                error_msg = "JSON error"
                try:
                    error_msg = resp.json().get("error", {}).get("message", "Unknown")
                except:
                    pass
                # If 4-band failed, fall back to 3-band PNG
                if bands == 4:
                    print(f"    4-band TIFF failed ({error_msg}), falling back to 3-band PNG...")
                    return fetch_naip_tile(bbox_proj, tile_size_px, bands=3)
                print(f"    Server error: {error_msg}")
                return None

            img = Image.open(BytesIO(resp.content))
            arr = np.array(img, dtype=np.uint8)

            # Handle different band configurations
            if arr.ndim == 2:
                # Grayscale — unlikely but handle it
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[2] == 4 and bands == 3:
                # Got RGBA when we wanted RGB — drop alpha
                arr = arr[:, :, :3]

            # Check if image is blank
            if arr.mean() < 5 or arr.mean() > 250:
                print(f"    Warning: Image appears blank (mean={arr.mean():.1f})")
                return None

            return arr

        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt + 1}/3), retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            # If TIFF parsing failed, fall back to PNG
            if bands == 4 and attempt == 0:
                print(f"    4-band TIFF error ({e}), falling back to 3-band PNG...")
                return fetch_naip_tile(bbox_proj, tile_size_px, bands=3)
            print(f"    Error (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    return None


def get_deepforest_model():
    """Load DeepForest model ONCE and reuse for all tiles."""
    global _deepforest_model
    if _deepforest_model is not None:
        return _deepforest_model

    try:
        from deepforest import main as dfmain
        _deepforest_model = dfmain.deepforest()
        # DeepForest 2.x auto-loads pre-trained weights, no use_release() needed
        print("  ✓ DeepForest model loaded successfully")
        return _deepforest_model
    except Exception as e:
        print(f"  ✗ DeepForest not available: {e}")
        print(f"    → Using green-channel vegetation mask as fallback")
        print(f"    → To fix: pip install deepforest==1.4.0 torch==2.1.0 torchvision==0.16.0")
        return None


def make_deepforest_mask(naip_image, model=None):
    """Run DeepForest on a NAIP tile to get tree-crown mask, with fallback.

    For large tiles (>400px), we manually split into 400×400 patches with
    10% overlap, run predict_image on each patch, then merge detections.
    This is necessary because predict_image() does NOT accept patch_size
    in DeepForest 2.x, and predict_tile() API varies across versions.

    Manual patching gives consistent results regardless of DeepForest version.
    """
    if model is not None:
        try:
            # DeepForest expects uint8 RGB (0-255)
            if naip_image.dtype != np.uint8:
                image_input = naip_image.astype(np.uint8)
            else:
                image_input = naip_image

            # Ensure 3-channel RGB
            if image_input.ndim == 3 and image_input.shape[2] > 3:
                image_input = image_input[:, :, :3]

            h, w = image_input.shape[:2]
            patch_size = 400
            overlap = 0.1
            stride = int(patch_size * (1 - overlap))  # 360px

            all_boxes = []

            if h > patch_size or w > patch_size:
                # Manual patch-based detection
                n_patches = 0
                for y0 in range(0, h, stride):
                    for x0 in range(0, w, stride):
                        y1 = min(y0 + patch_size, h)
                        x1 = min(x0 + patch_size, w)

                        # Skip tiny edge patches
                        if (y1 - y0) < 100 or (x1 - x0) < 100:
                            continue

                        patch = image_input[y0:y1, x0:x1].copy()
                        n_patches += 1

                        try:
                            boxes = model.predict_image(image=patch)
                        except Exception:
                            continue

                        if boxes is None or len(boxes) == 0:
                            continue

                        # Offset bounding boxes to full-image coordinates
                        boxes = boxes.copy()
                        boxes["xmin"] += x0
                        boxes["xmax"] += x0
                        boxes["ymin"] += y0
                        boxes["ymax"] += y0
                        all_boxes.append(boxes)

                if all_boxes:
                    import pandas as pd
                    all_boxes = pd.concat(all_boxes, ignore_index=True)
                else:
                    raise ValueError(f"No detections in {n_patches} patches")
            else:
                # Small image — single prediction
                all_boxes = model.predict_image(image=image_input)
                if all_boxes is None or len(all_boxes) == 0:
                    raise ValueError("No detections")

            # Filter by confidence
            all_boxes = all_boxes[all_boxes["score"] >= 0.3].reset_index(drop=True)

            if len(all_boxes) == 0:
                raise ValueError("No detections above score threshold 0.3")

            # Create binary mask from bounding boxes
            mask = np.zeros(naip_image.shape[:2], dtype=np.float32)
            n_trees = 0
            for _, row in all_boxes.iterrows():
                x1 = max(0, int(row["xmin"]))
                y1 = max(0, int(row["ymin"]))
                x2 = min(w, int(row["xmax"]))
                y2 = min(h, int(row["ymax"]))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 1.0
                    n_trees += 1

            return mask, f"DeepForest ({n_trees} trees)"

        except Exception as e:
            print(f"    DeepForest failed: {e} — using green-channel fallback")

    # Fallback: green-channel vegetation mask
    green = naip_image[:, :, 1].astype(np.float32)
    red = naip_image[:, :, 0].astype(np.float32)
    veg_mask = ((green > red) & (green > green.mean())).astype(np.float32)
    return veg_mask, "green-channel fallback"


def create_overlay_image(naip_image, original_poly_proj, aligned_poly_proj,
                         bbox_proj, idx, dca_code, shift_m, host, save_path):
    """Create an overlay showing NAIP + original polygon (red) + aligned polygon (green).

    The tile covers BOTH polygons with padding, so you can see the full shift.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=100)

    # Show NAIP as background
    ax.imshow(naip_image, extent=[bbox_proj[0], bbox_proj[2], bbox_proj[1], bbox_proj[3]])

    # Helper: plot shapely polygon (outline only — no fill so imagery is visible)
    def plot_polygon(ax, poly, color, linewidth=2.5):
        if poly.is_empty:
            return
        if poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                plot_polygon(ax, p, color, linewidth)
            return
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)

    # Plot ORIGINAL (misaligned) polygon in RED — outline only
    plot_polygon(ax, original_poly_proj, "red", linewidth=3.0)

    # Plot ALIGNED polygon in GREEN — outline only
    plot_polygon(ax, aligned_poly_proj, "lime", linewidth=3.0)

    # Plot centroids
    orig_cx, orig_cy = original_poly_proj.centroid.x, original_poly_proj.centroid.y
    align_cx, align_cy = aligned_poly_proj.centroid.x, aligned_poly_proj.centroid.y
    ax.plot(orig_cx, orig_cy, "rx", markersize=14, markeredgewidth=3)
    ax.plot(align_cx, align_cy, "g+", markersize=14, markeredgewidth=3)

    # Draw arrow from original to aligned centroid
    ax.annotate("", xy=(align_cx, align_cy), xytext=(orig_cx, orig_cy),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5))

    # Title with details
    extent_m = bbox_proj[2] - bbox_proj[0]
    ax.set_title(
        f"Polygon {idx} | DCA={dca_code} ({host})\n"
        f"Shift = {shift_m:.1f}m | Tile covers {extent_m:.0f}m × {extent_m:.0f}m",
        fontsize=13, fontweight="bold"
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="red", lw=3, label=f"Original (misaligned)"),
        plt.Line2D([0], [0], color="lime", lw=3, label=f"Aligned (corrected)"),
        plt.Line2D([0], [0], color="yellow", lw=2, label=f"Shift direction ({shift_m:.0f}m)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xlabel("Easting (m, EPSG:6557)")
    ax.set_ylabel("Northing (m, EPSG:6557)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def build_seed_data():
    """Main function: build paired samples from aligned + original polygons."""

    start_time = time.time()

    # ================================================================
    # Load data
    # ================================================================
    aligned_gdf, aligned_gdf_proj = load_aligned_polygons()
    obs_id_map = load_original_polygons()

    # Transformer: project CRS → WGS84 (only for diagnostic logging now)
    to_wgs84 = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

    # Create output directories
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    NAIP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try loading DeepForest once
    print("=" * 70)
    print("STEP 3: Loading DeepForest Model")
    print("=" * 70)
    df_model = get_deepforest_model()

    # ================================================================
    # Process each aligned polygon
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 4: Downloading NAIP Tiles & Building Paired Samples")
    print("=" * 70)
    print(f"  Server:           {NAIP_URL}")
    print(f"  Resolution:       {RESOLUTION_M}m (native, no downsampling)")
    print(f"  Min tile size:    {TRAINING_TILE_EXTENT_M}m × {TRAINING_TILE_EXTENT_M}m "
          f"({int(TRAINING_TILE_EXTENT_M / RESOLUTION_M)}px)")
    print(f"  Padding:          {OVERLAY_PADDING_M}m around both polygons")
    print()
    print(f"  For EACH polygon, we download ONE unified tile that:")
    print(f"    - Centers between aligned + original polygon midpoint")
    print(f"    - Covers BOTH polygons with generous padding")
    print(f"    - Used for training (.pkl), DeepForest masks, AND visualization")
    print()

    paired_samples = []
    failed = []
    link_stats = {"matched": 0, "not_found": 0, "no_notes": 0}
    training_download_times = []

    for idx, (_, aligned_row) in enumerate(aligned_gdf.iterrows()):
        aligned_row_proj = aligned_gdf_proj.iloc[idx]
        obs_id_raw = aligned_row.get("notes", "")

        # --- Check OBSERVATION_ID ---
        if not obs_id_raw or str(obs_id_raw) == "None" or str(obs_id_raw).strip() == "":
            print(f"  [{idx:>2d}] SKIP: No OBSERVATION_ID in notes column")
            failed.append({"idx": idx, "reason": "no_observation_id"})
            link_stats["no_notes"] += 1
            continue

        obs_id_clean = str(obs_id_raw).strip("{}")

        # --- Find original polygon ---
        if obs_id_clean not in obs_id_map:
            print(f"  [{idx:>2d}] SKIP: OBSERVATION_ID not found: ...{obs_id_clean[-12:]}")
            failed.append({"idx": idx, "reason": "obs_id_not_found", "obs_id": obs_id_clean})
            link_stats["not_found"] += 1
            continue

        link_stats["matched"] += 1
        original_row = obs_id_map[obs_id_clean]

        # --- Geometries in project CRS ---
        aligned_poly = aligned_row_proj.geometry
        original_poly = original_row.geometry

        # --- Compute shift ---
        shift_m = aligned_poly.centroid.distance(original_poly.centroid)

        dca_code = aligned_row.get("dca_code", "unknown")
        host = aligned_row.get("host", "unknown")

        print(f"  [{idx:>2d}] DCA={dca_code} ({host}) | Shift={shift_m:.1f}m")

        # =============================================================
        # UNIFIED TILE: covers BOTH original + aligned polygons
        # =============================================================
        # Strategy: center between both polygon centroids, expand to
        # cover both polygons with generous padding. This gives us ONE
        # tile that serves ALL purposes:
        #   - Training data (.pkl) — CNN sees both polygon positions
        #   - DeepForest mask — tree detection across the full area
        #   - Overlay visualization — before/after clearly visible
        # =============================================================

        # Center on midpoint between the two polygon centroids
        tile_cx = (aligned_poly.centroid.x + original_poly.centroid.x) / 2.0
        tile_cy = (aligned_poly.centroid.y + original_poly.centroid.y) / 2.0

        # Compute bounding box that covers BOTH polygons
        combined = unary_union([aligned_poly, original_poly])
        combined_bounds = combined.bounds  # (minx, miny, maxx, maxy)
        combined_width = combined_bounds[2] - combined_bounds[0]
        combined_height = combined_bounds[3] - combined_bounds[1]
        combined_max_dim = max(combined_width, combined_height)

        # Tile extent = max(minimum_extent, combined_polygons + padding)
        # Ensures both polygons fit comfortably with at least 150m buffer
        tile_extent_m = max(
            TRAINING_TILE_EXTENT_M,  # At least 800m
            combined_max_dim + 2 * OVERLAY_PADDING_M + 200  # Both polys + 200m extra
        )
        tile_buffer_m = tile_extent_m / 2.0
        tile_size_px = min(int(tile_extent_m / RESOLUTION_M), 3000)  # Cap at 3000px

        # Compute bbox in EPSG:6557 (meters)
        tile_bbox = (
            tile_cx - tile_buffer_m,
            tile_cy - tile_buffer_m,
            tile_cx + tile_buffer_m,
            tile_cy + tile_buffer_m,
        )

        # Diagnostic logging
        cx_wgs, cy_wgs = to_wgs84.transform(tile_cx, tile_cy)
        print(f"       Center (midpoint): EPSG:6557=({tile_cx:.1f}, {tile_cy:.1f})  "
              f"WGS84=({cx_wgs:.6f}, {cy_wgs:.6f})")
        print(f"       Tile extent:       {tile_extent_m:.0f}m × {tile_extent_m:.0f}m "
              f"({tile_size_px}px)")

        t0 = time.time()
        naip_image = fetch_naip_tile(tile_bbox, tile_size_px, bands=4)
        t_download = time.time() - t0
        training_download_times.append(t_download)

        if naip_image is None:
            print(f"       FAILED: Could not fetch NAIP tile")
            failed.append({"idx": idx, "reason": "tile_download_failed"})
            continue

        n_bands = naip_image.shape[2] if naip_image.ndim == 3 else 1

        # Save tile as TIFF (preserves 4 bands) or PNG (3 bands)
        if n_bands == 4:
            tile_path = NAIP_DIR / f"tile_{idx:02d}_DCA{dca_code}.tiff"
        else:
            tile_path = NAIP_DIR / f"tile_{idx:02d}_DCA{dca_code}.png"
        Image.fromarray(naip_image).save(tile_path)

        print(f"       NAIP tile:     {naip_image.shape[1]}×{naip_image.shape[0]}px "
              f"× {n_bands}bands ({tile_extent_m:.0f}m) in {t_download:.1f}s → {tile_path.name}")

        # =============================================================
        # Overlay visualization (using the same unified tile)
        # =============================================================
        # Use RGB for visualization
        naip_rgb_vis = naip_image[:, :, :3] if n_bands >= 3 else naip_image
        vis_path = VIS_DIR / f"overlay_{idx:02d}_DCA{dca_code}.png"
        try:
            create_overlay_image(
                naip_rgb_vis, original_poly, aligned_poly,
                tile_bbox, idx, dca_code, shift_m, host, vis_path
            )
            print(f"       Overlay vis:   {vis_path.name}")
        except Exception as e:
            print(f"       Overlay vis failed: {e}")

        # =============================================================
        # DeepForest mask (on the unified tile)
        # =============================================================
        # DeepForest needs RGB (3-channel), extract from 4-band if needed
        naip_rgb = naip_image[:, :, :3] if n_bands == 4 else naip_image
        deepforest_mask, mask_method = make_deepforest_mask(naip_rgb, model=df_model)
        coverage = deepforest_mask.mean()
        print(f"       Mask:          {mask_method} | coverage={coverage:.1%}")

        # --- Bundle the paired sample ---
        sample = {
            "naip_image": naip_image,                     # (H, W, 3-or-4) uint8
            "deepforest_mask": deepforest_mask,            # (H, W) float32
            "polygon": aligned_poly,                       # shapely Polygon (ALIGNED = ground truth)
            "polygon_original": original_poly,             # shapely Polygon (MISALIGNED)
            "bbox": tile_bbox,                             # (xmin, ymin, xmax, ymax) in project CRS
            "shift_meters": shift_m,
            "dca_code": dca_code,
            "host": host,
            "observation_id": obs_id_raw,
            "tile_size_px": tile_size_px,
            "tile_extent_m": tile_extent_m,
            "mask_method": mask_method,
        }
        paired_samples.append(sample)
        print(f"       ✓ OK")
        print()

    # ================================================================
    # SUMMARY STATISTICS
    # ================================================================
    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\n--- Overall ---")
    print(f"  Total aligned polygons:  {len(aligned_gdf)}")
    print(f"  Successfully processed:  {len(paired_samples)}")
    print(f"  Failed:                  {len(failed)}")
    print(f"  Total time:              {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(f"\n--- Linking ---")
    print(f"  Matched by OBSERVATION_ID:  {link_stats['matched']}")
    print(f"  No OBSERVATION_ID:          {link_stats['no_notes']}")
    print(f"  ID not found in GDB:        {link_stats['not_found']}")

    if paired_samples:
        shifts = [s["shift_meters"] for s in paired_samples]
        print(f"\n--- Shift Statistics (Original → Aligned) ---")
        print(f"  Mean shift:   {np.mean(shifts):.1f}m")
        print(f"  Median shift: {np.median(shifts):.1f}m")
        print(f"  Min shift:    {np.min(shifts):.1f}m")
        print(f"  Max shift:    {np.max(shifts):.1f}m")
        print(f"  Std dev:      {np.std(shifts):.1f}m")

        print(f"\n--- DCA Code Distribution (in seed data) ---")
        dca_counts = Counter(s["dca_code"] for s in paired_samples)
        for code, count in dca_counts.most_common():
            host = next(s["host"] for s in paired_samples if s["dca_code"] == code)
            print(f"  {code} ({host}): {count} samples")

        print(f"\n--- Tile Info ---")
        print(f"  Min extent:      {TRAINING_TILE_EXTENT_M:.0f}m × {TRAINING_TILE_EXTENT_M:.0f}m at 30cm")
        print(f"  Avg download:    {np.mean(training_download_times):.1f}s per tile")
        print(f"  Total downloads: {sum(training_download_times):.1f}s")

        print(f"\n--- Mask Statistics ---")
        coverages = [s["deepforest_mask"].mean() for s in paired_samples]
        methods = Counter(s["mask_method"] for s in paired_samples)
        for method, count in methods.items():
            print(f"  Method: {method} ({count} tiles)")
        print(f"  Avg coverage:   {np.mean(coverages):.1%}")
        print(f"  Min coverage:   {np.min(coverages):.1%}")
        print(f"  Max coverage:   {np.max(coverages):.1%}")

        # Memory estimate
        total_bytes = sum(
            s["naip_image"].nbytes + s["deepforest_mask"].nbytes
            for s in paired_samples
        )
        print(f"\n--- Storage ---")
        print(f"  Image data in memory: {total_bytes / 1024 / 1024:.1f} MB")

    # ================================================================
    # Save results
    # ================================================================
    if paired_samples:
        with open(OUTPUT_PKL, "wb") as f:
            pickle.dump(paired_samples, f)

        file_size = OUTPUT_PKL.stat().st_size
        n_naip_files = len(list(NAIP_DIR.glob("*.png")))
        n_overlay_files = len(list(VIS_DIR.glob("*.png")))

        print(f"\n{'=' * 70}")
        print(f"OUTPUT FILES")
        print(f"{'=' * 70}")
        print(f"  Paired samples (.pkl): {OUTPUT_PKL}")
        print(f"    → File size: {file_size / 1024 / 1024:.1f} MB")
        print(f"    → Contains {len(paired_samples)} samples")
        print(f"  NAIP tiles (separate): {NAIP_DIR}/ ({n_naip_files} images)")
        print(f"  Overlay images:        {VIS_DIR}/ ({n_overlay_files} images)")
        print(f"\n  ✓ Ready for weak augmentation engine!")
    else:
        print(f"\n  ✗ No samples produced. Check errors above.")

    if failed:
        print(f"\n--- Failed Polygons ---")
        for f_info in failed:
            print(f"  Polygon {f_info['idx']}: {f_info['reason']}")


if __name__ == "__main__":
    build_seed_data()
