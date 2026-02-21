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


def _residual_summary_norm(residual_flow_norm_bhwc: Tensor | None) -> dict[str, float]:
    if residual_flow_norm_bhwc is None:
        return {
            "residual_dx_abs_mean_norm": 0.0,
            "residual_dy_abs_mean_norm": 0.0,
            "residual_dx_abs_max_norm": 0.0,
            "residual_dy_abs_max_norm": 0.0,
        }

    if residual_flow_norm_bhwc.ndim != 4 or residual_flow_norm_bhwc.shape[-1] != 2:
        raise ValueError("residual_flow_norm_bhwc must have shape [B,H,W,2]")

    dx_abs = residual_flow_norm_bhwc[..., 0].abs()
    dy_abs = residual_flow_norm_bhwc[..., 1].abs()

    return {
        "residual_dx_abs_mean_norm": float(dx_abs.mean().item()),
        "residual_dy_abs_mean_norm": float(dy_abs.mean().item()),
        "residual_dx_abs_max_norm": float(dx_abs.max().item()),
        "residual_dy_abs_max_norm": float(dy_abs.max().item()),
    }


def evaluate_safety(
    grid: Tensor,
    residual_flow_norm_bhwc: Tensor | None = None,
    *,
    invalid_border_ratio: float | None = None,
    config: SafetyConfig | None = None,
) -> dict[str, Any]:
    """Evaluate runtime safety checks for a BHWC sampling grid.

    Notes:
    - `invalid_border_ratio` is currently a placeholder input. In this milestone,
      caller may provide it; otherwise it defaults to 0.0.
    """
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [B,H,W,2]")

    cfg = config or SafetyConfig()

    oob_ratio = _out_of_bounds_ratio(grid)
    border_ratio = 0.0 if invalid_border_ratio is None else float(invalid_border_ratio)
    jac = jacobian_stats(grid)
    residual_metrics = _residual_summary_norm(residual_flow_norm_bhwc)

    metrics = {
        "out_of_bounds_ratio": oob_ratio,
        "invalid_border_ratio": border_ratio,
        "negative_det_pct": float(jac["negative_det_pct"]),
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
