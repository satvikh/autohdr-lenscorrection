from __future__ import annotations

import torch
from torch import Tensor


def total_variation_loss(flow_bchw: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Total variation regularizer for flow fields.

    Args:
        flow_bchw: Tensor[B, 2, H, W] in pixel units.
    """
    if flow_bchw.ndim != 4 or flow_bchw.shape[1] != 2:
        raise ValueError("flow_bchw must have shape [B,2,H,W]")

    dx = flow_bchw[:, :, :, 1:] - flow_bchw[:, :, :, :-1]
    dy = flow_bchw[:, :, 1:, :] - flow_bchw[:, :, :-1, :]

    tv_x = torch.sqrt((dx * dx) + eps).mean()
    tv_y = torch.sqrt((dy * dy) + eps).mean()
    return tv_x + tv_y


def flow_magnitude_loss(flow_bchw: Tensor, *, eps: float = 1e-6) -> Tensor:
    if flow_bchw.ndim != 4 or flow_bchw.shape[1] != 2:
        raise ValueError("flow_bchw must have shape [B,2,H,W]")
    mag = torch.sqrt((flow_bchw * flow_bchw).sum(dim=1, keepdim=True) + eps)
    return mag.mean()


def flow_curvature_loss(flow_bchw: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Second-order smoothness regularizer."""
    if flow_bchw.ndim != 4 or flow_bchw.shape[1] != 2:
        raise ValueError("flow_bchw must have shape [B,2,H,W]")

    dxx = flow_bchw[:, :, :, 2:] - (2.0 * flow_bchw[:, :, :, 1:-1]) + flow_bchw[:, :, :, :-2]
    dyy = flow_bchw[:, :, 2:, :] - (2.0 * flow_bchw[:, :, 1:-1, :]) + flow_bchw[:, :, :-2, :]

    curv_x = torch.sqrt((dxx * dxx) + eps).mean() if dxx.numel() > 0 else torch.zeros((), device=flow_bchw.device, dtype=flow_bchw.dtype)
    curv_y = torch.sqrt((dyy * dyy) + eps).mean() if dyy.numel() > 0 else torch.zeros((), device=flow_bchw.device, dtype=flow_bchw.dtype)
    return curv_x + curv_y