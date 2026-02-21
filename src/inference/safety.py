from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.geometry.jacobian import jacobian_stats


@dataclass
class SafetyConfig:
    max_out_of_bounds_ratio: float = 0.01
    max_invalid_border_ratio: float = 0.05
    max_negative_det_pct: float = 0.0
    min_det_min: float = 0.0
    min_det_p01: float = 0.0
    max_residual_dx_abs_mean_norm: float = 0.25
    max_residual_dy_abs_mean_norm: float = 0.25
    max_residual_dx_abs_max_norm: float = 1.00
    max_residual_dy_abs_max_norm: float = 1.00


def _out_of_bounds_ratio(grid: Tensor) -> float:
    x = grid[..., 0]
    y = grid[..., 1]
    oob = (x < -1.0) | (x > 1.0) | (y < -1.0) | (y > 1.0)
    return float(oob.float().mean().item())


def _invalid_border_ratio_from_grid(grid: Tensor) -> float:
    """Estimate border risk as out-of-bounds ratio on the output border only."""
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [B,H,W,2]")

    x = grid[..., 0]
    y = grid[..., 1]
    oob = (x < -1.0) | (x > 1.0) | (y < -1.0) | (y > 1.0)

    _, h, w = oob.shape
    if h == 0 or w == 0:
        return 0.0

    border_mask = torch.zeros_like(oob, dtype=torch.bool)
    border_mask[:, 0, :] = True
    border_mask[:, -1, :] = True
    border_mask[:, :, 0] = True
    border_mask[:, :, -1] = True

    border_oob = oob & border_mask
    border_count = int(border_mask.sum().item())
    if border_count == 0:
        return 0.0
    return float(border_oob.sum().item()) / float(border_count)


def _residual_summary_norm(residual_flow_norm_bhwc: Tensor | None) -> dict[str, float]:
    if residual_flow_norm_bhwc is None:
        return {
            "residual_dx_abs_mean_norm": 0.0,
            "residual_dy_abs_mean_norm": 0.0,
            "residual_dx_abs_max_norm": 0.0,
            "residual_dy_abs_max_norm": 0.0,
            "residual_magnitude": 0.0,
        }

    if residual_flow_norm_bhwc.ndim != 4 or residual_flow_norm_bhwc.shape[-1] != 2:
        raise ValueError("residual_flow_norm_bhwc must have shape [B,H,W,2]")

    dx_abs = residual_flow_norm_bhwc[..., 0].abs()
    dy_abs = residual_flow_norm_bhwc[..., 1].abs()
    mag = torch.sqrt((dx_abs * dx_abs) + (dy_abs * dy_abs))

    return {
        "residual_dx_abs_mean_norm": float(dx_abs.mean().item()),
        "residual_dy_abs_mean_norm": float(dy_abs.mean().item()),
        "residual_dx_abs_max_norm": float(dx_abs.max().item()),
        "residual_dy_abs_max_norm": float(dy_abs.max().item()),
        "residual_magnitude": float(mag.max().item()),
    }


def evaluate_safety(
    grid: Tensor,
    residual_flow_norm_bhwc: Tensor | None = None,
    *,
    invalid_border_ratio: float | None = None,
    config: SafetyConfig | None = None,
) -> dict[str, Any]:
    """Evaluate runtime safety checks for a BHWC sampling grid."""
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [B,H,W,2]")

    cfg = config or SafetyConfig()

    oob_ratio = _out_of_bounds_ratio(grid)
    if invalid_border_ratio is None:
        border_ratio = _invalid_border_ratio_from_grid(grid)
    else:
        border_ratio = float(invalid_border_ratio)
    jac = jacobian_stats(grid)
    residual_metrics = _residual_summary_norm(residual_flow_norm_bhwc)

    negative_det_pct = float(jac["negative_det_pct"])
    metrics = {
        # Legacy keys used by current tooling/tests.
        "out_of_bounds_ratio": oob_ratio,
        "invalid_border_ratio": border_ratio,
        "negative_det_pct": negative_det_pct,
        # Contract aliases.
        "oob_ratio": oob_ratio,
        "border_invalid_ratio": border_ratio,
        "jacobian_negative_det_pct": negative_det_pct,
        # Shared Jacobian diagnostics.
        "det_min": float(jac["det_min"]),
        "det_p01": float(jac["det_p01"]),
        "det_mean": float(jac["det_mean"]),
        "high_grad_area_frac": float(jac.get("high_grad_area_frac", 0.0)),
        **residual_metrics,
    }

    reasons: list[str] = []
    if metrics["out_of_bounds_ratio"] > cfg.max_out_of_bounds_ratio:
        reasons.append("OOB_RATIO_EXCEEDED")
    if metrics["invalid_border_ratio"] > cfg.max_invalid_border_ratio:
        reasons.append("INVALID_BORDER_RATIO_EXCEEDED")
    if metrics["negative_det_pct"] > cfg.max_negative_det_pct:
        reasons.append("JACOBIAN_FOLDOVER_PCT_EXCEEDED")
    if metrics["det_min"] < cfg.min_det_min:
        reasons.append("JACOBIAN_DET_MIN_TOO_LOW")
    if metrics["det_p01"] < cfg.min_det_p01:
        reasons.append("JACOBIAN_DET_P01_TOO_LOW")

    if metrics["residual_dx_abs_mean_norm"] > cfg.max_residual_dx_abs_mean_norm:
        reasons.append("RESIDUAL_DX_MEAN_EXCEEDED")
    if metrics["residual_dy_abs_mean_norm"] > cfg.max_residual_dy_abs_mean_norm:
        reasons.append("RESIDUAL_DY_MEAN_EXCEEDED")
    if metrics["residual_dx_abs_max_norm"] > cfg.max_residual_dx_abs_max_norm:
        reasons.append("RESIDUAL_DX_MAX_EXCEEDED")
    if metrics["residual_dy_abs_max_norm"] > cfg.max_residual_dy_abs_max_norm:
        reasons.append("RESIDUAL_DY_MAX_EXCEEDED")

    return {
        "safe": len(reasons) == 0,
        "reasons": reasons,
        "metrics": metrics,
    }
