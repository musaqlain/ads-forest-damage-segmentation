"""Polygon rasterisation and NAIP image normalisation.

  - rasterize_polygon       shapely Polygon -> binary numpy mask
  - rasterize_polygon_soft  same, Gaussian-blurred for smooth optimisation
  - normalize_naip          uint8 NAIP image -> float32 tensor
  - make_rasterio_transform Affine transform for a tile bounding box
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping, Polygon
import rasterio.features
from rasterio.transform import from_bounds


def make_rasterio_transform(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
):
    """Create a rasterio Affine transform mapping pixel coords → geographic coords.

    This transform tells rasterio: "pixel (0,0) corresponds to geographic
    coordinate (xmin, ymax), and each pixel step is (dx, dy) in CRS units."

    Args:
        bbox: (xmin, ymin, xmax, ymax) in project CRS.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        rasterio.Affine transform object.
    """
    return from_bounds(
        west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3],
        width=width, height=height
    )


def rasterize_polygon(
    polygon: Polygon,
    image_shape: tuple[int, int],
    transform,
) -> np.ndarray:
    """Rasterize a single shapely Polygon to a binary mask.

    The output mask has value 1.0 inside the polygon and 0.0 outside.

    Args:
        polygon: A shapely Polygon (or MultiPolygon) in the same CRS as transform.
        image_shape: (height, width) of the output mask.
        transform: rasterio.Affine mapping pixels → geographic coords.

    Returns:
        np.ndarray of shape (H, W), dtype float32, values in {0.0, 1.0}.
    """
    if polygon is None or polygon.is_empty:
        return np.zeros(image_shape, dtype=np.float32)

    mask = rasterio.features.rasterize(
        shapes=[(mapping(polygon), 1.0)],
        out_shape=image_shape,
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        all_touched=False,  # Only pixels with centroid inside polygon
    )
    return mask


def rasterize_polygon_soft(
    polygon: Polygon,
    image_shape: tuple[int, int],
    transform,
    sigma: float = 2.0,
) -> np.ndarray:
    """Rasterize a polygon with Gaussian blur for smooth gradient signal.

    Rationale:
      A hard binary mask has zero gradient everywhere except exactly at the
      boundary pixel. During training, if the predicted alignment is off by
      10+ pixels, the gradient signal from the binary mask is zero, giving
      the optimizer no direction to move.

      By applying a small Gaussian blur (σ=2 ≈ 3 pixel blur radius), the
      transition zone around the polygon boundary becomes ~6 pixels wide.
      The optimizer can "feel" the direction toward correct alignment even
      when the prediction is several pixels off. This is used ONLY during
      the early training phases; inference uses the hard mask.

    Args:
        polygon: shapely Polygon.
        image_shape: (H, W).
        transform: rasterio.Affine.
        sigma: Gaussian blur radius in pixels. 2.0 = ~6px effective boundary.

    Returns:
        np.ndarray (H, W), float32, values in [0.0, 1.0].
    """
    hard_mask = rasterize_polygon(polygon, image_shape, transform)
    soft_mask = gaussian_filter(hard_mask, sigma=sigma)
    # Normalize to [0, 1] in case the Gaussian extends values
    soft_mask = np.clip(soft_mask, 0.0, 1.0)
    return soft_mask.astype(np.float32)


def normalize_naip(image: np.ndarray) -> torch.Tensor:
    """Normalize a NAIP image to a float32 tensor in [0, 1].

    Handles both 3-band (RGB) and 4-band (RGBN) images.

    Args:
        image: np.ndarray of shape (H, W, C), dtype uint8. C can be 3 or 4.

    Returns:
        torch.Tensor of shape (C, H, W), dtype float32, values in [0, 1].
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW


def resize_to_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an image (H, W, C) to (size, size, C) using PIL for quality.

    Used to standardize NAIP tiles to config.tile_size_px before model input.
    """
    from PIL import Image
    pil = Image.fromarray(image)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.array(pil)
