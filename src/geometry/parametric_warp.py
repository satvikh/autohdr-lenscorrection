from __future__ import annotations

import torch
from torch import Tensor

from src.geometry.coords import make_identity_grid

# Starter safety ranges from the spec.
K1_RANGE = (-0.6, 0.6)
K2_RANGE = (-0.3, 0.3)
K3_RANGE = (-0.15, 0.15)
P_RANGE = (-0.03, 0.03)
DC_RANGE = (-0.08, 0.08)
S_RANGE = (0.90, 1.20)
ASPECT_RANGE = (0.97, 1.03)

# Feature flag: optional aspect term is disabled by default.
ENABLE_ASPECT = False


def _clamp_channel(values: Tensor, value_range: tuple[float, float]) -> Tensor:
    return values.clamp(min=value_range[0], max=value_range[1])


def _unpack_and_clamp_params(params: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if params.ndim != 2:
        raise ValueError("params must have shape [B, N]")

    if params.shape[1] < 8:
        raise ValueError("params must have at least 8 channels: [k1,k2,k3,p1,p2,dcx,dcy,s]")

    k1 = _clamp_channel(params[:, 0], K1_RANGE)
    k2 = _clamp_channel(params[:, 1], K2_RANGE)
    k3 = _clamp_channel(params[:, 2], K3_RANGE)
    p1 = _clamp_channel(params[:, 3], P_RANGE)
    p2 = _clamp_channel(params[:, 4], P_RANGE)
    dcx = _clamp_channel(params[:, 5], DC_RANGE)
    dcy = _clamp_channel(params[:, 6], DC_RANGE)
    s = _clamp_channel(params[:, 7], S_RANGE)

    if ENABLE_ASPECT and params.shape[1] >= 9:
        aspect = _clamp_channel(params[:, 8], ASPECT_RANGE)
    else:
        aspect = torch.ones_like(k1)

    return k1, k2, k3, p1, p2, dcx, dcy, s, aspect


def build_parametric_grid(
    params: Tensor,
    height: int,
    width: int,
    align_corners: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build a BHWC backward sampling grid from Brown-Conrady parameters.

    Signature and shape are fixed by the geometry contract:
    build_parametric_grid(...) -> Tensor[B,H,W,2]
    """
    if align_corners is not True:
        raise ValueError("align_corners must be True per global geometry policy")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    params_t = params.to(device=device, dtype=dtype)
    batch = params_t.shape[0]

    k1, k2, k3, p1, p2, dcx, dcy, s, aspect = _unpack_and_clamp_params(params_t)

    base_grid = make_identity_grid(batch, height, width, device=device, dtype=dtype)
    x_out = base_grid[..., 0]
    y_out = base_grid[..., 1]

    dcx = dcx.view(batch, 1, 1)
    dcy = dcy.view(batch, 1, 1)
    s = s.view(batch, 1, 1)

    # Coordinates centered around principal-point offset.
    x = (x_out - dcx) / s
    y = (y_out - dcy) / s

    if ENABLE_ASPECT:
        x = x / aspect.view(batch, 1, 1)

    r2 = (x * x) + (y * y)
    r4 = r2 * r2
    r6 = r4 * r2

    k1 = k1.view(batch, 1, 1)
    k2 = k2.view(batch, 1, 1)
    k3 = k3.view(batch, 1, 1)
    p1 = p1.view(batch, 1, 1)
    p2 = p2.view(batch, 1, 1)

    radial = 1.0 + (k1 * r2) + (k2 * r4) + (k3 * r6)

    x_tan = (2.0 * p1 * x * y) + (p2 * (r2 + 2.0 * x * x))
    y_tan = (p1 * (r2 + 2.0 * y * y)) + (2.0 * p2 * x * y)

    x_src = (x * radial) + x_tan
    y_src = (y * radial) + y_tan

    if ENABLE_ASPECT:
        x_src = x_src * aspect.view(batch, 1, 1)

    x_src = x_src + dcx
    y_src = y_src + dcy

    return torch.stack((x_src, y_src), dim=-1)
