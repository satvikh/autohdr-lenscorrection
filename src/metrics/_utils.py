"""Internal helpers for metric preprocessing and numeric stability."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy_image(image: Any) -> np.ndarray:
    """Convert an input image-like object to HWC float32 in [0, 1]."""
    if isinstance(image, np.ndarray):
        arr = image
    else:
        # Optional torch support without importing torch unconditionally.
        if hasattr(image, "detach") and hasattr(image, "cpu") and hasattr(image, "numpy"):
            arr = image.detach().cpu().numpy()
        elif hasattr(image, "__array__"):
            arr = np.asarray(image)
        else:
            raise TypeError("Unsupported image type")

    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3:
        # Convert CHW -> HWC if needed.
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Expected 2D/3D image, got shape {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    max_v = float(np.nanmax(arr)) if arr.size else 1.0
    if max_v > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert HWC image to HW grayscale in [0, 1]."""
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.shape[-1] >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return np.mean(arr, axis=-1)


def _ensure_same_shape(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raise if prediction and target shapes do not match exactly."""
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    return pred, gt


def _safe_float(value: float) -> float:
    """Clamp non-finite values to 0 for robust scoring."""
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))
