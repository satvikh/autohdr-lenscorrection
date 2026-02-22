from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _as_float_image_pair(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shape, got {pred.shape} vs {target.shape}")
    if pred.ndim != 4:
        raise ValueError("pred and target must have shape [B,C,H,W]")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError("pred and target must be floating point tensors")
    # Always run gradient losses in FP32 for numeric stability under autocast / mixed precision.
    return pred.float(), target.float()


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
    pred_f, target_f = _as_float_image_pair(pred, target)

    pred_gx, pred_gy = sobel_gradients(pred_f)
    tgt_gx, tgt_gy = sobel_gradients(target_f)
    pred_mag = gradient_magnitude(pred_gx, pred_gy, eps=eps)
    tgt_mag = gradient_magnitude(tgt_gx, tgt_gy, eps=eps)
    out = (pred_mag - tgt_mag).abs().mean()
    out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=1e6)
    if not torch.isfinite(out):
        raise RuntimeError("edge_magnitude_loss produced non-finite value")
    return out


def gradient_orientation_cosine_loss(
    pred: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-6,
    weight_by_target_magnitude: bool = True,
    target_weight_cap: float = 5.0,
) -> Tensor:
    """Gradient orientation mismatch loss.

    Cosine similarity is computed between gradient vectors (gx, gy).
    Loss = 1 - cosine similarity, optionally weighted by GT magnitude.
    """
    pred_f, target_f = _as_float_image_pair(pred, target)

    pred_gx, pred_gy = sobel_gradients(pred_f)
    tgt_gx, tgt_gy = sobel_gradients(target_f)

    dot = (pred_gx * tgt_gx) + (pred_gy * tgt_gy)
    pred_norm = torch.sqrt((pred_gx * pred_gx) + (pred_gy * pred_gy) + eps)
    tgt_norm = torch.sqrt((tgt_gx * tgt_gx) + (tgt_gy * tgt_gy) + eps)

    denom = torch.clamp(pred_norm * tgt_norm, min=eps)
    cos = dot / denom
    cos = torch.clamp(cos, min=-1.0, max=1.0)
    base = torch.nan_to_num(1.0 - cos, nan=0.0, posinf=2.0, neginf=2.0)

    if not weight_by_target_magnitude:
        out = base.mean()
        if not torch.isfinite(out):
            raise RuntimeError("gradient_orientation_cosine_loss produced non-finite value")
        return out

    denom_w = torch.clamp(tgt_norm.mean(), min=eps)
    weights = tgt_norm / denom_w
    weights = torch.nan_to_num(weights, nan=0.0, posinf=target_weight_cap, neginf=0.0)
    if target_weight_cap > 0.0:
        weights = torch.clamp(weights, min=0.0, max=float(target_weight_cap))
    out = (base * weights).mean()
    if not torch.isfinite(out):
        raise RuntimeError("gradient_orientation_cosine_loss produced non-finite value")
    return out


def _to_luma(image: Tensor) -> Tensor:
    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W]")
    if image.shape[1] == 1:
        return image
    if image.shape[1] >= 3:
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]
        return (0.299 * r) + (0.587 * g) + (0.114 * b)
    return image.mean(dim=1, keepdim=True)


def _orientation_histogram_from_gradients(
    gx: Tensor,
    gy: Tensor,
    *,
    bins: int,
    eps: float,
    magnitude_power: float,
    mask: Tensor | None = None,
) -> Tensor:
    if gx.shape != gy.shape:
        raise ValueError("gx and gy must have identical shape")
    if gx.ndim != 4 or gx.shape[1] != 1:
        raise ValueError("gx and gy must have shape [B,1,H,W]")
    if bins <= 1:
        raise ValueError("bins must be > 1")

    mag = torch.sqrt((gx * gx) + (gy * gy) + eps)
    theta = torch.remainder(torch.atan2(gy, gx), torch.pi)

    if mask is not None:
        if mask.shape != mag.shape:
            raise ValueError("mask must have same shape as gradient tensors")
        weights = torch.pow(torch.clamp(mag, min=eps), float(magnitude_power)) * mask
    else:
        weights = torch.pow(torch.clamp(mag, min=eps), float(magnitude_power))

    b, _, h, w = theta.shape
    n = h * w
    theta_flat = theta.view(b, n)
    w_flat = weights.view(b, n)

    pos = theta_flat / torch.pi * float(bins)
    idx0 = torch.floor(pos).to(dtype=torch.long) % bins
    frac = torch.clamp(pos - torch.floor(pos), min=0.0, max=1.0)
    idx1 = (idx0 + 1) % bins
    w0 = (1.0 - frac) * w_flat
    w1 = frac * w_flat

    hist = torch.zeros((b, bins), device=gx.device, dtype=gx.dtype)
    hist.scatter_add_(1, idx0, w0)
    hist.scatter_add_(1, idx1, w1)

    hist = hist / torch.clamp(hist.sum(dim=1, keepdim=True), min=eps)
    return hist


def line_orientation_hist_loss(
    pred: Tensor,
    target: Tensor,
    *,
    bins: int = 36,
    eps: float = 1e-6,
    magnitude_power: float = 1.0,
    target_mag_quantile: float = 0.70,
    target_mag_temperature: float = 0.05,
) -> Tensor:
    """Line-orientation histogram mismatch surrogate.

    This is a differentiable line-straightness proxy:
    - convert to luminance
    - compute Sobel orientations
    - build weighted orientation histograms in [0, pi)
    - minimize (1 - cosine(hist_pred, hist_target))
    """
    pred_f, target_f = _as_float_image_pair(pred, target)
    pred_l = _to_luma(pred_f)
    target_l = _to_luma(target_f)

    pred_gx, pred_gy = sobel_gradients(pred_l)
    tgt_gx, tgt_gy = sobel_gradients(target_l)

    tgt_mag = gradient_magnitude(tgt_gx, tgt_gy, eps=eps)
    q = float(max(min(target_mag_quantile, 1.0), 0.0))
    if q > 0.0:
        thr = torch.quantile(tgt_mag.detach().reshape(tgt_mag.shape[0], -1), q=q, dim=1)
        thr = thr.view(-1, 1, 1, 1)
        temp = max(float(target_mag_temperature), eps)
        line_mask = torch.sigmoid((tgt_mag - thr) / temp)
    else:
        line_mask = torch.ones_like(tgt_mag)

    hist_pred = _orientation_histogram_from_gradients(
        pred_gx,
        pred_gy,
        bins=int(bins),
        eps=eps,
        magnitude_power=float(magnitude_power),
        mask=line_mask,
    )
    hist_tgt = _orientation_histogram_from_gradients(
        tgt_gx,
        tgt_gy,
        bins=int(bins),
        eps=eps,
        magnitude_power=float(magnitude_power),
        mask=line_mask,
    )

    dot = (hist_pred * hist_tgt).sum(dim=1)
    denom = torch.clamp(torch.linalg.vector_norm(hist_pred, dim=1) * torch.linalg.vector_norm(hist_tgt, dim=1), min=eps)
    cos = torch.clamp(dot / denom, min=-1.0, max=1.0)
    out = torch.nan_to_num(1.0 - cos, nan=0.0, posinf=2.0, neginf=2.0).mean()
    if not torch.isfinite(out):
        raise RuntimeError("line_orientation_hist_loss produced non-finite value")
    return out


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
