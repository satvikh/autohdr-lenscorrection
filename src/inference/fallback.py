from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


@dataclass
class ConservativeClampConfig:
    k1: tuple[float, float] = (-0.3, 0.3)
    k2: tuple[float, float] = (-0.15, 0.15)
    k3: tuple[float, float] = (-0.08, 0.08)
    p: tuple[float, float] = (-0.015, 0.015)
    dc: tuple[float, float] = (-0.04, 0.04)
    s: tuple[float, float] = (0.95, 1.05)


SafetyEvaluator = Callable[[Tensor, str], dict[str, Any]]


def make_conservative_param_only(
    params: Tensor,
    config: ConservativeClampConfig | None = None,
) -> Tensor:
    """Apply tighter runtime clamps for conservative fallback mode."""
    if params.ndim != 2 or params.shape[1] < 8:
        raise ValueError("params must have shape [B,8+]")

    cfg = config or ConservativeClampConfig()
    out = params.clone()

    out[:, 0] = out[:, 0].clamp(*cfg.k1)
    out[:, 1] = out[:, 1].clamp(*cfg.k2)
    out[:, 2] = out[:, 2].clamp(*cfg.k3)
    out[:, 3] = out[:, 3].clamp(*cfg.p)
    out[:, 4] = out[:, 4].clamp(*cfg.p)
    out[:, 5] = out[:, 5].clamp(*cfg.dc)
    out[:, 6] = out[:, 6].clamp(*cfg.dc)
    out[:, 7] = out[:, 7].clamp(*cfg.s)
    return out


def run_fallback_hierarchy(
    hybrid_params: Tensor,
    param_only_params: Tensor,
    safety_evaluator: SafetyEvaluator,
    clamp_config: ConservativeClampConfig | None = None,
) -> tuple[str, Tensor, list[str], dict[str, Any]]:
    """Run required fallback order: hybrid -> param_only -> conservative param_only."""
    warnings: list[str] = []

    report_hybrid = safety_evaluator(hybrid_params, "hybrid")
    if report_hybrid.get("safe", False):
        return "hybrid", hybrid_params, warnings, report_hybrid
    warnings.append("HYBRID_UNSAFE_FALLBACK_TO_PARAM_ONLY")

    report_param = safety_evaluator(param_only_params, "param_only")
    if report_param.get("safe", False):
        return "param_only", param_only_params, warnings, report_param
    warnings.append("PARAM_ONLY_UNSAFE_FALLBACK_TO_CONSERVATIVE")

    conservative = make_conservative_param_only(param_only_params, clamp_config)
    report_cons = safety_evaluator(conservative, "param_only_conservative")
    if report_cons.get("safe", False):
        return "param_only_conservative", conservative, warnings, report_cons

    warnings.append("HARD_UNSAFE_OUTPUT")
    return "param_only_conservative", conservative, warnings, report_cons
