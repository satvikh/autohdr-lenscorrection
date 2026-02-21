from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0  # 4:4:4
JPEG_OPTIMIZE = False
JPEG_PROGRESSIVE = False


def _tensor_to_uint8_hwc(image: Tensor) -> np.ndarray:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image tensor must have shape [1, C, H, W]")

    image = image.detach().cpu().to(dtype=torch.float32)
    image = image.clamp(0.0, 1.0)

    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    elif image.shape[1] != 3:
        raise ValueError("image tensor must have 1 or 3 channels")

    chw = image[0]
    hwc = chw.permute(1, 2, 0).numpy()
    return (hwc * 255.0 + 0.5).astype(np.uint8)


def save_jpeg(image: Tensor, output_path: str | Path, expected_hw: tuple[int, int] | None = None) -> tuple[int, int]:
    """Save a tensor image to deterministic JPEG and validate dimensions.

    Args:
        image: Tensor with shape [1, C, H, W], float in [0,1].
        output_path: Destination file path.
        expected_hw: Optional (H, W) expected output dimensions.

    Returns:
        Saved image dimensions as (H, W).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _tensor_to_uint8_hwc(image)
    h, w = int(arr.shape[0]), int(arr.shape[1])

    if expected_hw is not None and (h, w) != expected_hw:
        raise ValueError(f"output tensor dimensions {(h, w)} do not match expected {expected_hw}")

    pil_img = Image.fromarray(arr, mode="RGB")
    pil_img.save(
        out_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=JPEG_OPTIMIZE,
        progressive=JPEG_PROGRESSIVE,
    )

    with Image.open(out_path) as check_img:
        check_rgb = check_img.convert("RGB")
        saved_w, saved_h = check_rgb.size

    if (saved_h, saved_w) != (h, w):
        raise RuntimeError(
            f"saved JPEG dimensions {(saved_h, saved_w)} do not match tensor dimensions {(h, w)}"
        )

    return saved_h, saved_w
