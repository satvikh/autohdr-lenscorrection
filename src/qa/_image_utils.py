"""Internal image helpers for QA checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image_rgb_float(path: str | Path) -> np.ndarray:
    """Load an image file as HWC float32 RGB in [0, 1].

    Raises:
        ValueError: If the image cannot be decoded.
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

