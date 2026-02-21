"""Internal image I/O and normalization utilities for proxy metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_image_rgb_float(path: str | Path) -> np.ndarray:
    """Load an image from disk as HWC float32 RGB in [0, 1].

    Raises:
        ValueError: If Pillow is unavailable or the image cannot be loaded.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError("Failed to load image: Pillow is not installed") from exc

    path_obj = Path(path)
    try:
        with Image.open(path_obj) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Failed to load image '{path_obj}': {exc}") from exc

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Failed to load image '{path_obj}': expected HWC RGB, got {arr.shape}")
    return arr


def to_rgb_float_array(image: Any) -> np.ndarray:
    """Convert array/tensor-like image input into HWC float32 RGB in [0, 1]."""
    if isinstance(image, (str, Path)):
        return load_image_rgb_float(image)

    if isinstance(image, np.ndarray):
        arr = image
    elif hasattr(image, "detach") and hasattr(image, "cpu") and hasattr(image, "numpy"):
        arr = image.detach().cpu().numpy()
    elif hasattr(image, "__array__"):
        arr = np.asarray(image)
    else:
        raise ValueError(f"Unsupported image input type: {type(image)!r}")

    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    max_v = float(np.nanmax(arr)) if arr.size else 1.0
    if max_v > 1.0:
        arr = arr / 255.0

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape[-1]}")

    return np.clip(arr, 0.0, 1.0)

