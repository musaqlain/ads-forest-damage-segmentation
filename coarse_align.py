"""
Stage 1: ProximityAlign-Inspired Alignment for ADS Polygon Correction
======================================================================

Adapted from Cherif et al. (ISPRS 2024) "Novel Approaches for Aligning
Geospatial Vector Maps" — modified for forest damage (diffuse boundaries)
rather than urban buildings (crisp edges).

Approach:
  1. Compute vegetation stress from NAIP (NGRDI + NDVI + brightness anomaly)
  2. Build proximity map from stress signal boundaries (distance transform)
  3. Score candidate translations using hybrid energy:
     a) Inside-outside stress contrast (polygon interior vs surroundings)
     b) Contour-to-boundary distance (ProximityAlign core energy)
     c) Out-of-bounds penalty
  4. Coarse-to-fine grid search for optimal (tx, ty)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    uniform_filter, distance_transform_edt, binary_dilation,
    binary_erosion, gaussian_filter
)
from typing import Optional, Tuple


class CoarseAligner:
    """ProximityAlign-inspired alignment for ADS polygon correction.

    Parameters:
        search_radius_px: Maximum displacement to search (pixels).
        coarse_step: Step size for coarse search pass.
        fine_radius: Radius around coarse best for fine search.
        damage_threshold: Percentile for damage boundary extraction.
        energy_weights: (contrast_w, contour_w, oob_w) weights.
        min_mask_pixels: Minimum polygon pixels to attempt alignment.
    """

    def __init__(
        self,
        search_radius_px: int = 250,
        coarse_step: int = 4,
        fine_radius: int = 8,
        damage_threshold: float = 65.0,
        energy_weights: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        min_mask_pixels: int = 50,
    ) -> None:
        self.search_radius = search_radius_px
        self.coarse_step = coarse_step
        self.fine_radius = fine_radius
        self.damage_threshold = damage_threshold
        self.energy_weights = energy_weights
        self.min_mask_pixels = min_mask_pixels

    # ================================================================
    # PUBLIC API
    # ================================================================

    def align(
        self,
        naip_image: np.ndarray,
        polygon_mask: np.ndarray,
        deepforest_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """Find optimal (tx, ty) to align polygon mask with damage signal.

        Returns: (tx, ty, score) where score is 0-1 confidence.
        """
        H, W = naip_image.shape[:2]

        if (polygon_mask > 0.5).sum() < self.min_mask_pixels:
            return 0.0, 0.0, 0.0

        # Build signal maps
        stress_map = self._compute_stress_map(naip_image)
        damage_map = self._compute_damage_map(naip_image, deepforest_mask)
        proximity_map = self._compute_proximity_map(stress_map)

        # Extract polygon geometry
        contour_ys, contour_xs = self._extract_contour(polygon_mask)
        if len(contour_ys) < 10:
            return 0.0, 0.0, 0.0

        # Interior pixels (subsampled for speed)
        int_ys, int_xs = np.where(polygon_mask > 0.5)
        if len(int_ys) > 500:
            step_sub = max(1, len(int_ys) // 500)
            int_ys, int_xs = int_ys[::step_sub], int_xs[::step_sub]

        global_mean = float(np.mean(stress_map))

        # Multi-scale search
        tx, ty, energy = self._search(
            stress_map, proximity_map,
            contour_ys, contour_xs,
            int_ys, int_xs,
            global_mean, H, W
        )

        # Energy -> confidence score
        score = max(0.0, min(1.0, 0.5 - energy))
        return tx, ty, score

    def get_diagnostic_maps(
        self,
        naip_image: np.ndarray,
        polygon_mask: np.ndarray,
        deepforest_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Return intermediate maps for visualization."""
        stress_map = self._compute_stress_map(naip_image)
        damage_map = self._compute_damage_map(naip_image, deepforest_mask)
        proximity_map = self._compute_proximity_map(stress_map)

        contour_ys, contour_xs = self._extract_contour(polygon_mask)
        contour_mask = np.zeros_like(polygon_mask)
        if len(contour_ys) > 0:
            contour_mask[contour_ys, contour_xs] = 1.0

        return {
            "stress_map": stress_map,
            "damage_map": damage_map,
            "proximity_map": proximity_map,
            "contour_mask": contour_mask,
            "gap_map": self._compute_gap_map(deepforest_mask, naip_image)
                       if deepforest_mask is not None else None,
        }

    # ================================================================
    # SIGNAL COMPUTATION
    # ================================================================

    @staticmethod
    def _compute_stress_map(naip_image: np.ndarray) -> np.ndarray:
        """Vegetation stress from NAIP imagery. Returns (H,W) in [0,1]."""
        img = naip_image.astype(np.float32)
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        has_nir = img.shape[2] >= 4

        # NGRDI
        ngrdi = (R - G) / (R + G + 1.0)
        stress_ngrdi = np.clip((ngrdi + 1.0) / 2.0 - 0.4, 0.0, 0.6) / 0.6

        if has_nir:
            NIR = img[:, :, 3]
            ndvi = (NIR - R) / (NIR + R + 1.0)
            dead_boost = np.clip((0.2 - ndvi) / 0.15, 0.0, 1.0)
            stress_spectral = 0.7 * stress_ngrdi + 0.3 * dead_boost
        else:
            stress_spectral = stress_ngrdi

        # Brightness anomaly
        brightness = (R + G + B) / 3.0
        local_mean = uniform_filter(brightness, size=31)
        local_var = uniform_filter(brightness ** 2, size=31) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 1.0))
        anomaly = np.clip(np.abs(brightness - local_mean) / local_std / 3.0, 0.0, 1.0)

        if has_nir:
            stress = 0.8 * stress_spectral + 0.2 * anomaly
        else:
            stress = 0.7 * stress_spectral + 0.3 * anomaly

        return stress.astype(np.float32)

    @staticmethod
    def _compute_gap_map(
        deepforest_mask: np.ndarray, naip_image: np.ndarray
    ) -> np.ndarray:
        """DeepForest gap map (1 = gap/no tree, 0 = tree crown)."""
        gap = (1.0 - deepforest_mask).astype(np.float32)
        img = naip_image.astype(np.float32)
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        green_ratio = G / (R + G + B + 1.0)
        veg_mask = np.clip(uniform_filter((green_ratio > 0.33).astype(np.float32), size=15), 0, 1)
        return (gap * veg_mask).astype(np.float32)

    def _compute_damage_map(
        self, naip_image: np.ndarray, deepforest_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Combined damage likelihood. Returns (H,W) in [0,1]."""
        stress = self._compute_stress_map(naip_image)
        if deepforest_mask is not None:
            gap = self._compute_gap_map(deepforest_mask, naip_image)
            return (0.5 * stress + 0.5 * gap).astype(np.float32)
        return stress

    def _compute_proximity_map(self, signal_map: np.ndarray) -> np.ndarray:
        """Distance transform from signal boundaries. 0 = on edge."""
        positive = signal_map[signal_map > 0.01]
        if len(positive) < 100:
            return np.ones_like(signal_map) * 50.0

        threshold = np.percentile(positive, self.damage_threshold)
        binary = (signal_map > threshold).astype(bool)
        binary = binary_erosion(binary, iterations=1)
        binary = binary_dilation(binary, iterations=1)

        dilated = binary_dilation(binary, iterations=1)
        eroded = binary_erosion(binary, iterations=1)
        boundary = (dilated & ~binary) | (binary & ~eroded)

        if boundary.sum() < 10:
            return np.ones_like(signal_map) * 50.0

        dist = distance_transform_edt(~boundary)
        return gaussian_filter(dist.astype(np.float32), sigma=2.0)

    # ================================================================
    # GEOMETRY
    # ================================================================

    @staticmethod
    def _extract_contour(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract boundary pixels of a binary mask."""
        binary = (mask > 0.5).astype(bool)
        if binary.sum() < 4:
            return np.array([], dtype=int), np.array([], dtype=int)
        eroded = binary_erosion(binary, iterations=1)
        contour = binary & ~eroded
        return np.where(contour)

    # ================================================================
    # ENERGY FUNCTION
    # ================================================================

    def _energy(
        self,
        stress_map: np.ndarray,
        proximity_map: np.ndarray,
        contour_ys: np.ndarray,
        contour_xs: np.ndarray,
        int_ys: np.ndarray,
        int_xs: np.ndarray,
        global_mean: float,
        tx: int,
        ty: int,
        H: int,
        W: int,
    ) -> float:
        """Hybrid energy: inside-outside contrast + contour-edge distance.

        Lower = better alignment.
        """
        # --- Shift interior ---
        si_y, si_x = int_ys + ty, int_xs + tx
        ib_int = (si_y >= 0) & (si_y < H) & (si_x >= 0) & (si_x < W)
        n_int = ib_int.sum()
        if n_int < max(10, len(int_ys) * 0.3):
            return 999.0

        # A) Inside-outside stress contrast
        damage_in = float(np.mean(stress_map[si_y[ib_int], si_x[ib_int]]))
        contrast_energy = -(damage_in - global_mean)

        # --- Shift contour ---
        sc_y, sc_x = contour_ys + ty, contour_xs + tx
        ib_con = (sc_y >= 0) & (sc_y < H) & (sc_x >= 0) & (sc_x < W)
        n_con = ib_con.sum()
        if n_con < max(10, len(contour_ys) * 0.3):
            return 999.0

        # B) Contour-to-boundary distance
        dists = proximity_map[sc_y[ib_con], sc_x[ib_con]]
        contour_energy = float(np.mean(dists)) / 20.0

        # C) Out-of-bounds penalty
        oob = max(
            (len(int_ys) - n_int) / len(int_ys),
            (len(contour_ys) - n_con) / len(contour_ys)
        )

        a, b, c = self.energy_weights
        return a * contrast_energy + b * contour_energy + c * oob * 2.0

    # ================================================================
    # SEARCH
    # ================================================================

    def _search(
        self,
        stress_map: np.ndarray,
        proximity_map: np.ndarray,
        contour_ys: np.ndarray,
        contour_xs: np.ndarray,
        int_ys: np.ndarray,
        int_xs: np.ndarray,
        global_mean: float,
        H: int,
        W: int,
    ) -> Tuple[float, float, float]:
        """Two-pass coarse-to-fine search."""
        r = self.search_radius
        best_e, best_tx, best_ty = 999.0, 0, 0

        # Pass 1: coarse
        step = self.coarse_step
        for ty in range(-r, r + 1, step):
            for tx in range(-r, r + 1, step):
                e = self._energy(
                    stress_map, proximity_map,
                    contour_ys, contour_xs,
                    int_ys, int_xs, global_mean,
                    tx, ty, H, W
                )
                if e < best_e:
                    best_e, best_tx, best_ty = e, tx, ty

        # Pass 2: fine
        fr = self.fine_radius
        cx, cy = best_tx, best_ty
        for ty in range(cy - fr, cy + fr + 1):
            for tx in range(cx - fr, cx + fr + 1):
                if abs(tx) > r or abs(ty) > r:
                    continue
                e = self._energy(
                    stress_map, proximity_map,
                    contour_ys, contour_xs,
                    int_ys, int_xs, global_mean,
                    tx, ty, H, W
                )
                if e < best_e:
                    best_e, best_tx, best_ty = e, tx, ty

        return float(best_tx), float(best_ty), best_e

    # ================================================================
    # UTILITY
    # ================================================================

    @staticmethod
    def shift_mask(mask: np.ndarray, tx: float, ty: float) -> np.ndarray:
        """Translate a binary mask by (tx, ty) pixels."""
        dx, dy = int(round(tx)), int(round(ty))
        H, W = mask.shape
        shifted = np.zeros_like(mask)

        src_y_lo, src_y_hi = max(0, -dy), min(H, H - dy)
        src_x_lo, src_x_hi = max(0, -dx), min(W, W - dx)
        dst_y_lo, dst_y_hi = max(0, dy), min(H, H + dy)
        dst_x_lo, dst_x_hi = max(0, dx), min(W, W + dx)

        h = min(src_y_hi - src_y_lo, dst_y_hi - dst_y_lo)
        w = min(src_x_hi - src_x_lo, dst_x_hi - dst_x_lo)

        if h > 0 and w > 0:
            shifted[dst_y_lo:dst_y_lo + h, dst_x_lo:dst_x_lo + w] = \
                mask[src_y_lo:src_y_lo + h, src_x_lo:src_x_lo + w]

        return shifted
