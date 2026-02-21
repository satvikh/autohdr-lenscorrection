"""Composite proxy score entrypoint and aggregation helpers."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from src.metrics.hardfail_checks import run_hardfail_checks
from src.metrics.proxy_edge import compute_edge_score
from src.metrics.proxy_gradient import compute_gradient_score
from src.metrics.proxy_line import compute_line_score
from src.metrics.proxy_ssim_mae import compute_mae, compute_ssim

_METRIC_KEYS = ("edge", "line", "grad", "ssim", "mae")


def _weights(config) -> Dict[str, float]:
    cfg = config or {}
    defaults = {"edge": 0.40, "line": 0.22, "grad": 0.18, "ssim": 0.15, "mae": 0.05}
    custom = cfg.get("weights", {}) if isinstance(cfg, dict) else {}
    merged = {**defaults, **custom}
    total = sum(float(v) for v in merged.values())
    if total <= 0:
        return defaults
    return {k: float(v) / total for k, v in merged.items()}


def _enabled_metrics(config) -> dict[str, bool]:
    cfg = config or {}
    mcfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else {}
    return {k: bool(mcfg.get(k, True)) for k in _METRIC_KEYS}


def _metric_functions() -> dict[str, Callable]:
    return {
        "edge": compute_edge_score,
        "line": compute_line_score,
        "grad": compute_gradient_score,
        "ssim": compute_ssim,
        "mae": compute_mae,  # Converted to mae_score after compute.
    }


def _hardfail_total(config) -> float:
    cfg = config or {}
    acfg = cfg.get("aggregation", {}) if isinstance(cfg, dict) else {}
    fail_policy = str(acfg.get("fail_policy", "exclude"))
    if fail_policy == "score_zero":
        return 0.0
    if fail_policy == "score_neg_inf":
        return -1e9
    return float("nan")


def compute_proxy_score(pred, gt, config) -> dict:
    """Compute total proxy score and per-component diagnostics.

    Returns:
        {"total": float, "subscores": dict[str, float], "flags": {"hardfail": bool, "reasons": list[str]}}
    """
    cfg = config or {}
    hardfail, reasons = run_hardfail_checks(pred, gt, cfg)
    if hardfail:
        return {
            "total": _hardfail_total(cfg),
            "subscores": {k: float("nan") for k in _METRIC_KEYS},
            "flags": {"hardfail": True, "reasons": reasons},
        }

    metric_fns = _metric_functions()
    enabled = _enabled_metrics(cfg)

    subscores = {k: float("nan") for k in _METRIC_KEYS}
    for key, is_enabled in enabled.items():
        if not is_enabled:
            continue
        value = float(metric_fns[key](pred, gt, cfg))
        if key == "mae":
            value = float(np.clip(1.0 - np.clip(value, 0.0, 1.0), 0.0, 1.0))
        subscores[key] = float(np.clip(value, 0.0, 1.0))

    enabled_keys = [k for k, is_enabled in enabled.items() if is_enabled]
    if not enabled_keys:
        total = float("nan")
    else:
        w = _weights(cfg)
        w_enabled = {k: float(w.get(k, 0.0)) for k in enabled_keys}
        w_sum = float(sum(w_enabled.values()))
        if w_sum <= 0.0:
            uniform = 1.0 / float(len(enabled_keys))
            w_enabled = {k: uniform for k in enabled_keys}
        else:
            w_enabled = {k: v / w_sum for k, v in w_enabled.items()}
        total = float(sum(w_enabled[k] * subscores[k] for k in enabled_keys))

    return {
        "total": float(np.clip(total, 0.0, 1.0)) if np.isfinite(total) else float(total),
        "subscores": subscores,
        "flags": {"hardfail": False, "reasons": reasons},
    }


def aggregate_scores(rows: List[dict], config) -> dict:
    """Aggregate per-image proxy rows into a compact summary dict.

    `rows` may include additional keys (for example `image_id`) supplied by callers.
    """
    if not rows:
        return {
            "count": 0,
            "fail_count": 0,
            "mean_total": 0.0,
            "mean_subscores": {"edge": 0.0, "line": 0.0, "grad": 0.0, "ssim": 0.0, "mae": 0.0},
        }

    cfg = config or {}
    acfg = cfg.get("aggregation", {}) if isinstance(cfg, dict) else {}
    fail_policy = str(acfg.get("fail_policy", "exclude"))

    totals = np.array([float(r.get("total", float("nan"))) for r in rows], dtype=np.float64)
    fail_count = int(sum(bool(r.get("flags", {}).get("hardfail", False)) for r in rows))
    if fail_policy == "exclude":
        finite_totals = totals[np.isfinite(totals)]
    else:
        finite_totals = totals
    mean_total = float(np.mean(finite_totals)) if finite_totals.size else float("nan")

    means = {}
    for k in _METRIC_KEYS:
        vals = np.array([float(r.get("subscores", {}).get(k, float("nan"))) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        means[k] = float(np.mean(vals)) if vals.size else float("nan")

    return {
        "count": len(rows),
        "fail_count": fail_count,
        "mean_total": mean_total,
        "mean_subscores": means,
    }
