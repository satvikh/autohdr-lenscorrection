from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def adapt_residual_flow_to_bhwc(residual_flow: Tensor) -> Tensor:
    """Accept BCHW or BHWC residual flow and return BHWC float tensor.

    Accepted layouts:
    - BCHW: [B, 2, Hr, Wr]
    - BHWC: [B, Hr, Wr, 2]
    """
    if residual_flow.ndim != 4:
        raise ValueError("residual_flow must be 4D and in BCHW or BHWC layout")

    if residual_flow.shape[1] == 2:
        # BCHW -> BHWC
        flow = residual_flow.permute(0, 2, 3, 1).contiguous()
    elif residual_flow.shape[-1] == 2:
        # Already BHWC
        flow = residual_flow.contiguous()
    else:
        raise ValueError("residual_flow must be BCHW [B,2,H,W] or BHWC [B,H,W,2]")

    if not torch.is_floating_point(flow):
        flow = flow.float()

    return flow


def pixel_flow_to_normalized_delta(flow_bhwc_px: Tensor) -> Tensor:
    """Convert pixel displacement BHWC flow to normalized grid delta (align_corners=True)."""
    if flow_bhwc_px.ndim != 4 or flow_bhwc_px.shape[-1] != 2:
        raise ValueError("flow_bhwc_px must have shape [B,H,W,2]")

    b, hr, wr, _ = flow_bhwc_px.shape
    flow = flow_bhwc_px

    dx = flow[..., 0]
    dy = flow[..., 1]

    if wr > 1:
        dx_norm = (2.0 * dx) / float(wr - 1)
    else:
        dx_norm = torch.zeros_like(dx)

    if hr > 1:
        dy_norm = (2.0 * dy) / float(hr - 1)
    else:
        dy_norm = torch.zeros_like(dy)

    out = torch.stack((dx_norm, dy_norm), dim=-1)
    assert out.shape == (b, hr, wr, 2)
    return out


def adapt_and_normalize_residual_flow(residual_flow: Tensor) -> Tensor:
    """Adapter utility: BCHW/BHWC pixel flow -> canonical BHWC normalized delta."""
    flow_bhwc_px = adapt_residual_flow_to_bhwc(residual_flow)
    return pixel_flow_to_normalized_delta(flow_bhwc_px)


def upsample_residual_flow(
    flow_lr: Tensor,
    target_h: int,
    target_w: int,
    align_corners: bool = True,
) -> Tensor:
    """Upsample normalized BHWC residual flow to full resolution (BHWC)."""
    if align_corners is not True:
        raise ValueError("align_corners must be True per global geometry policy")
    if flow_lr.ndim != 4 or flow_lr.shape[-1] != 2:
        raise ValueError("flow_lr must have shape [B,H,W,2]")
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_h and target_w must be positive")

    # BHWC -> BCHW for interpolate.
    flow_bchw = flow_lr.permute(0, 3, 1, 2).contiguous()
    up_bchw = F.interpolate(
        flow_bchw,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=True,
    )
    up_bhwc = up_bchw.permute(0, 2, 3, 1).contiguous()
    return up_bhwc


def fuse_grids(param_grid: Tensor, residual_flow: Tensor) -> Tensor:
    """Fuse full-resolution normalized residual delta into parametric grid.

    Computes: G_final = G_param + Delta_G_residual
    """
    if param_grid.ndim != 4 or param_grid.shape[-1] != 2:
        raise ValueError("param_grid must have shape [B,H,W,2]")
    if residual_flow.ndim != 4 or residual_flow.shape[-1] != 2:
        raise ValueError("residual_flow must have shape [B,H,W,2]")
    if param_grid.shape != residual_flow.shape:
        raise ValueError("param_grid and residual_flow must have identical shape")

    if not torch.is_floating_point(param_grid):
        param_grid = param_grid.float()
    if not torch.is_floating_point(residual_flow):
        residual_flow = residual_flow.float()

    return param_grid + residual_flow
