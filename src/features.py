"""ROI cropping, grid colour features and noise-based augmentation.

The feature representation is the one actually used by the published scripts:
a 3x3 grid of per-cell RGB means (27 numbers), not a raw image tensor.
"""

from __future__ import annotations

import cv2
import numpy as np


def crop_roi(img_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the colorimetric pad and convert BGR -> RGB."""
    x1, y1, x2, y2 = roi
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb[y1:y2, x1:x2]


def grid_features(roi: np.ndarray, grid: int = 3, normalise: bool = True) -> np.ndarray:
    """Split the ROI into ``grid`` x ``grid`` cells and take per-cell RGB means."""
    h, w = roi.shape[:2]
    cell_h = h // grid
    cell_w = w // grid

    feats: list[float] = []
    for i in range(grid):
        for j in range(grid):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < grid - 1 else h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < grid - 1 else w
            cell = roi[y1:y2, x1:x2]
            feats.extend(
                [cell[:, :, 0].mean(), cell[:, :, 1].mean(), cell[:, :, 2].mean()]
            )

    arr = np.asarray(feats, dtype=np.float64)
    return arr / 255.0 if normalise else arr


def add_gaussian_noise(roi: np.ndarray, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise to a uint8 ROI and clip back to [0, 255]."""
    noise = np.random.normal(0, sigma, roi.shape)
    noisy = np.clip(roi.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def random_noise_roi(height: int, width: int) -> np.ndarray:
    """Uniform random colour ROI, used as a non-pad ('abnormal') stress sample."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def white_roi(height: int, width: int) -> np.ndarray:
    """Blank white ROI, used by the magnesium protocol."""
    return np.full((height, width, 3), 255, dtype=np.uint8)
