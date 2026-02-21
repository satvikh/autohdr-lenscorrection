from __future__ import annotations

import torch
from torch import Tensor


def _finite_diff_x(t: Tensor, step_x: float) -> Tensor:
    d = (t[:, :, 1:] - t[:, :, :-1]) / step_x
    return torch.cat((d, d[:, :, -1:]), dim=2)


def _finite_diff_y(t: Tensor, step_y: float) -> Tensor:
    d = (t[:, 1:, :] - t[:, :-1, :]) / step_y
    return torch.cat((d, d[:, -1:, :]), dim=1)


def jacobian_foldover_penalty(
    final_grid_bhwc: Tensor,
    *,
    margin: float = 0.0,
) -> Tensor:
    """Differentiable penalty for negative or near-negative Jacobian determinant.

    Args:
        final_grid_bhwc: Tensor[B, H, W, 2] containing sampling coordinates.
        margin: determinant margin threshold. Values below margin are penalized.
    """
    if final_grid_bhwc.ndim != 4 or final_grid_bhwc.shape[-1] != 2:
        raise ValueError("final_grid_bhwc must have shape [B,H,W,2]")

    b, h, w, _ = final_grid_bhwc.shape
    if h < 2 or w < 2:
        return torch.zeros((), device=final_grid_bhwc.device, dtype=final_grid_bhwc.dtype)

    step_x = 2.0 / float(w - 1)
    step_y = 2.0 / float(h - 1)

    x_src = final_grid_bhwc[..., 0]
    y_src = final_grid_bhwc[..., 1]

    dx_dx = _finite_diff_x(x_src, step_x)
    dx_dy = _finite_diff_y(x_src, step_y)
    dy_dx = _finite_diff_x(y_src, step_x)
    dy_dy = _finite_diff_y(y_src, step_y)

    det = (dx_dx * dy_dy) - (dx_dy * dy_dx)
    penalty = torch.relu(float(margin) - det)
    return penalty.mean()