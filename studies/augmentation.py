"""Synthetic affine displacement of polygons, with exactly known ground truth.

`SyntheticDisplacer` takes verified (polygon, image) pairs and produces unlimited
training pairs by applying known random affine transforms. Because the transform is
known, every generated pair has a perfect label.

Used by the affine-recovery studies in this folder to answer "which perturbations
can a network recover, and up to what magnitude?". The alignment approach those
studies evaluated is retired (see coarse_align.py), but the displacement engine
itself is still useful for corrupt-and-recover pretraining.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from shapely import affinity
from shapely.geometry import Polygon


class SyntheticDisplacer:
    """Generate random affine displacements for self-supervised training.

    The affine transform is sampled uniformly within displacement bounds
    set by the training curriculum (large → medium → small displacements).

    Example:
        displacer = SyntheticDisplacer(max_translation_px=50, max_rotation_deg=10.0)
        affine_matrix = displacer.generate_random_affine()  # (2, 3) numpy
        displaced_polygon = displacer.displace_polygon(polygon, affine_matrix)
    """

    def __init__(
        self,
        max_translation_px: float = 150.0,
        max_rotation_deg: float = 15.0,
        scale_range: tuple[float, float] = (0.80, 1.20),
        max_shear_deg: float = 10.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.max_trans = max_translation_px
        self.max_rot = max_rotation_deg
        self.scale_range = scale_range
        self.max_shear = max_shear_deg
        self.rng = rng or np.random.default_rng()

    def generate_random_affine(self) -> np.ndarray:
        """Sample a random 2x3 affine matrix in PIXEL coordinate space.

        Composes all 6 affine degrees of freedom:
            tx, ty       -- translation (pixels)
            theta        -- rotation angle (degrees)
            sx, sy       -- independent anisotropic scale
            shear (phi)  -- skew angle (degrees)

        Full composition: A = R(theta) @ Shear(phi) @ S(sx, sy)
        This matches the order used in He et al. (KDD 2022) Eq. 1.

        Returns:
            np.ndarray of shape (2, 3), dtype float32.
        """
        theta_deg = self.rng.uniform(-self.max_rot, self.max_rot)
        theta_rad = theta_deg * np.pi / 180.0
        sx = self.rng.uniform(*self.scale_range)
        sy = self.rng.uniform(*self.scale_range)
        tx = self.rng.uniform(-self.max_trans, self.max_trans)
        ty = self.rng.uniform(-self.max_trans, self.max_trans)
        phi_deg = self.rng.uniform(-self.max_shear, self.max_shear)
        phi_rad = phi_deg * np.pi / 180.0

        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

        # Scale matrix S(sx, sy)
        S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float32)
        # Shear matrix: [[1, tan(phi)], [0, 1]]
        Sh = np.array([[1.0, np.tan(phi_rad)], [0.0, 1.0]], dtype=np.float32)
        # Rotation matrix R(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        # Full linear part: R @ Sh @ S
        A = R @ Sh @ S   # (2, 2)

        affine = np.array(
            [[A[0, 0], A[0, 1], tx],
             [A[1, 0], A[1, 1], ty]],
            dtype=np.float32,
        )
        return affine

    def displace_polygon(self, polygon: Polygon, affine_px: np.ndarray) -> Polygon:
        """Apply pixel-space affine to a shapely Polygon.

        Note: shapely.affinity.affine_transform expects parameters in the order:
          [a, b, d, e, xoff, yoff]  for  x' = a*x + b*y + xoff, y' = d*x + e*y + yoff

        This matches our affine_px layout (row 0 = x equation, row 1 = y equation).
        """
        a, b, xoff = float(affine_px[0, 0]), float(affine_px[0, 1]), float(affine_px[0, 2])
        d, e, yoff = float(affine_px[1, 0]), float(affine_px[1, 1]), float(affine_px[1, 2])
        return affinity.affine_transform(polygon, [a, b, d, e, xoff, yoff])

    def displace_mask_tensor(
        self,
        mask_tensor: torch.Tensor,
        affine_px: np.ndarray,
    ) -> torch.Tensor:
        """Apply affine to a rasterized mask using PyTorch grid_sample.

        The affine is in PIXEL coordinates; we must convert to STN's normalized
        [-1, 1] coordinate system before calling affine_grid.

        Normalization:
            x_norm = x_px / (W/2) - 1   →  tx_norm = tx_px / (W/2)
            The rotation/scale part (A matrix) is already unit-less but we
            scale by the ratio of H vs W to handle non-square images.

        Args:
            mask_tensor: (1, H, W) float32 binary mask tensor.
            affine_px:   (2, 3) affine in pixel coordinate space.

        Returns:
            (1, H, W) warped mask tensor.
        """
        H, W = mask_tensor.shape[1], mask_tensor.shape[2]

        # Convert from pixel-space affine to normalized-space affine
        # For STN: normalized coords span [-1, 1] ↔ pixel coords span [0, W] and [0, H]
        norm_affine = affine_px.copy()
        norm_affine[0, 2] = affine_px[0, 2] / (W / 2.0)  # tx: pixels → normalized
        norm_affine[1, 2] = affine_px[1, 2] / (H / 2.0)  # ty: pixels → normalized

        theta = torch.from_numpy(norm_affine).unsqueeze(0)   # (1, 2, 3)
        mask_4d = mask_tensor.unsqueeze(0)                   # (1, 1, H, W)
        grid = F.affine_grid(theta, mask_4d.size(), align_corners=False)
        warped = F.grid_sample(mask_4d, grid, align_corners=False, padding_mode="zeros")
        return warped.squeeze(0)  # (1, H, W)

    def to_normalized_affine(
        self, affine_px: np.ndarray, H: int, W: int
    ) -> torch.Tensor:
        """Convert pixel-space affine to STN normalized affine tensor.

        This is used to convert the GT affine to a tensor for the GridLoss.

        Returns:
            torch.Tensor (2, 3), the affine in normalized [-1,1] space.
        """
        norm_affine = affine_px.copy()
        norm_affine[0, 2] = affine_px[0, 2] / (W / 2.0)
        norm_affine[1, 2] = affine_px[1, 2] / (H / 2.0)
        return torch.from_numpy(norm_affine)
