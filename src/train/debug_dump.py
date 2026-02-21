from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def _to_uint8_hwc(image_bchw: Tensor) -> np.ndarray:
    if image_bchw.ndim != 4 or image_bchw.shape[0] < 1:
        raise ValueError("image_bchw must have shape [B,C,H,W] with B>=1")

    img = image_bchw[0].detach().cpu().float().clamp(0.0, 1.0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[0] != 3:
        raise ValueError("image channels must be 1 or 3")

    arr = img.permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def dump_debug_triplet(
    *,
    out_dir: str | Path,
    prefix: str,
    input_image: Tensor,
    pred_image: Tensor,
    target_image: Tensor,
) -> None:
    """Dump first-sample input/pred/target images for lightweight debug."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(_to_uint8_hwc(input_image), mode="RGB").save(out / f"{prefix}_input.png")
    Image.fromarray(_to_uint8_hwc(pred_image), mode="RGB").save(out / f"{prefix}_pred.png")
    Image.fromarray(_to_uint8_hwc(target_image), mode="RGB").save(out / f"{prefix}_target.png")