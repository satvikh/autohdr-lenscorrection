from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _sobel_kernels(*, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return kx, ky


def sobel_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    """Compute per-channel Sobel gradients.

    Args:
        image: Tensor[B, C, H, W]

    Returns:
        gx: Tensor[B, C, H, W]
        gy: Tensor[B, C, H, W]
    """
    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W]")

    b, c, _, _ = image.shape
    kx, ky = _sobel_kernels(device=image.device, dtype=image.dtype)
    kx = kx.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
    ky = ky.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()

    gx = F.conv2d(image, kx, bias=None, stride=1, padding=1, groups=c)
    gy = F.conv2d(image, ky, bias=None, stride=1, padding=1, groups=c)
    assert gx.shape == (b, c, image.shape[2], image.shape[3])
    assert gy.shape == (b, c, image.shape[2], image.shape[3])
    return gx, gy


def gradient_magnitude(gx: Tensor, gy: Tensor, *, eps: float = 1e-6) -> Tensor:
    if gx.shape != gy.shape:
        raise ValueError("gx and gy must have identical shape")
    return torch.sqrt((gx * gx) + (gy * gy) + eps)


def edge_magnitude_loss(pred: Tensor, target: Tensor, *, eps: float = 1e-6) -> Tensor:
    """L1 difference between Sobel gradient magnitudes."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shape, got {pred.shape} vs {target.shape}")

    pred_gx, pred_gy = sobel_gradients(pred)
    tgt_gx, tgt_gy = sobel_gradients(target)
    pred_mag = gradient_magnitude(pred_gx, pred_gy, eps=eps)
    tgt_mag = gradient_magnitude(tgt_gx, tgt_gy, eps=eps)
    return (pred_mag - tgt_mag).abs().mean()


def gradient_orientation_cosine_loss(
    pred: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-6,
    weight_by_target_magnitude: bool = True,
) -> Tensor:
    """Gradient orientation mismatch loss.

    Cosine similarity is computed between gradient vectors (gx, gy).
    Loss = 1 - cosine similarity, optionally weighted by GT magnitude.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shape, got {pred.shape} vs {target.shape}")

    pred_gx, pred_gy = sobel_gradients(pred)
    tgt_gx, tgt_gy = sobel_gradients(target)

    dot = (pred_gx * tgt_gx) + (pred_gy * tgt_gy)
    pred_norm = torch.sqrt((pred_gx * pred_gx) + (pred_gy * pred_gy) + eps)
    tgt_norm = torch.sqrt((tgt_gx * tgt_gx) + (tgt_gy * tgt_gy) + eps)

    cos = dot / (pred_norm * tgt_norm + eps)
    cos = torch.clamp(cos, min=-1.0, max=1.0)
    base = 1.0 - cos

    if not weight_by_target_magnitude:
        return base.mean()

    weights = tgt_norm / (tgt_norm.mean() + eps)
    return (base * weights).mean()


def resize_pair_explicit(pred: Tensor, target: Tensor, *, scale: float) -> tuple[Tensor, Tensor]:
    """Explicitly resize prediction and target to a common scale.

    This helper is intentionally explicit for multi-scale losses.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shape, got {pred.shape} vs {target.shape}")
    if pred.ndim != 4:
        raise ValueError("pred and target must have shape [B,C,H,W]")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    if scale == 1.0:
        return pred, target

    h = max(int(round(pred.shape[-2] * scale)), 1)
    w = max(int(round(pred.shape[-1] * scale)), 1)

    pred_s = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)
    tgt_s = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=True)
    return pred_s, tgt_s


def multiscale_average(
    pred: Tensor,
    target: Tensor,
    *,
    scales: Sequence[float],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Apply an image loss over explicit scales and average."""
    if len(scales) == 0:
        raise ValueError("scales must be non-empty")

    vals: list[Tensor] = []
    for scale in scales:
        pred_s, tgt_s = resize_pair_explicit(pred, target, scale=float(scale))
        vals.append(loss_fn(pred_s, tgt_s))

    stacked = torch.stack(vals)
    return stacked.mean()
