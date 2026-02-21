from __future__ import annotations

import torch
from torch import Tensor


def _finite_diff_x(t: Tensor, step_x: float) -> Tensor:
    # Forward difference with last-column replication.
    d = (t[:, :, 1:] - t[:, :, :-1]) / step_x
    return torch.cat((d, d[:, :, -1:]), dim=2)


def _finite_diff_y(t: Tensor, step_y: float) -> Tensor:
    # Forward difference with last-row replication.
    d = (t[:, 1:, :] - t[:, :-1, :]) / step_y
    return torch.cat((d, d[:, -1:, :]), dim=1)


def jacobian_stats(grid: Tensor) -> dict:
    """Compute Jacobian determinant statistics for BHWC warp grid.

    grid: Tensor[B,H,W,2] with (x_src, y_src) sampling coordinates.
    """
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [B,H,W,2]")

    grid_f = grid.float()
    b, h, w, _ = grid_f.shape

    if h < 2 or w < 2:
        det = torch.ones((b, h, w), device=grid_f.device, dtype=grid_f.dtype)
    else:
        step_x = 2.0 / float(w - 1)
        step_y = 2.0 / float(h - 1)

        x_src = grid_f[..., 0]
        y_src = grid_f[..., 1]

        dx_dx = _finite_diff_x(x_src, step_x)
        dx_dy = _finite_diff_y(x_src, step_y)
        dy_dx = _finite_diff_x(y_src, step_x)
        dy_dy = _finite_diff_y(y_src, step_y)

        det = (dx_dx * dy_dy) - (dx_dy * dy_dx)

    det_flat = det.reshape(-1)

    negative_det_pct = 100.0 * float((det_flat < 0.0).float().mean().item())
    det_min = float(det_flat.min().item())
    det_p01 = float(torch.quantile(det_flat, 0.01).item())
    det_mean = float(det_flat.mean().item())

    # Optional high-gradient area fraction: ||J - I||_F > threshold.
    if h < 2 or w < 2:
        high_grad_area_frac = 0.0
    else:
        j11 = _finite_diff_x(grid_f[..., 0], 2.0 / float(w - 1))
        j12 = _finite_diff_y(grid_f[..., 0], 2.0 / float(h - 1))
        j21 = _finite_diff_x(grid_f[..., 1], 2.0 / float(w - 1))
        j22 = _finite_diff_y(grid_f[..., 1], 2.0 / float(h - 1))

        frob = torch.sqrt((j11 - 1.0) ** 2 + j12**2 + j21**2 + (j22 - 1.0) ** 2)
        high_grad_area_frac = float((frob > 0.5).float().mean().item())

    out = {
        "negative_det_pct": negative_det_pct,
        "det_min": det_min,
        "det_p01": det_p01,
        "det_mean": det_mean,
        "high_grad_area_frac": high_grad_area_frac,
    }

    for value in out.values():
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError("jacobian_stats produced non-finite output")

    return out
