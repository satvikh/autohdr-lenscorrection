from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import Tensor

from src.train.protocols import ProxyScorer


def resolve_proxy_scorer(
    *,
    module_path: str | None,
    function_name: str = "compute_proxy_score",
) -> ProxyScorer | None:
    """Resolve optional proxy scorer callable.

    Returns None if module/callable is unavailable.
    """
    if not module_path:
        return None

    try:
        module = importlib.import_module(module_path)
    except Exception:
        return None

    fn = getattr(module, function_name, None)
    if not callable(fn):
        return None
    return fn  # type: ignore[return-value]


def _call_proxy_scorer(
    scorer: ProxyScorer,
    pred: Tensor,
    target: Tensor,
    config: Any | None,
) -> dict[str, Any]:
    return scorer(pred, target, config)


def compute_proxy_metrics_for_batch(
    *,
    scorer: ProxyScorer,
    pred_batch: Tensor,
    target_batch: Tensor,
    config: Any | None,
) -> dict[str, float]:
    """Compute batch-level proxy metrics with resilient fallback strategy.

    Strategy:
    1) Try scorer on full batch tensors.
    2) If that fails, run per-sample and average.
    """
    if pred_batch.shape != target_batch.shape:
        raise ValueError(f"pred_batch and target_batch must have same shape, got {pred_batch.shape} vs {target_batch.shape}")

    report: dict[str, Any] | None = None
    try:
        report = _call_proxy_scorer(scorer, pred_batch, target_batch, config)
    except Exception:
        report = None

    if report is None:
        totals: list[float] = []
        sub_acc: dict[str, list[float]] = {}

        for i in range(pred_batch.shape[0]):
            item_report = _call_proxy_scorer(scorer, pred_batch[i : i + 1], target_batch[i : i + 1], config)
            total_score = item_report.get("total_score")
            if isinstance(total_score, (int, float)):
                totals.append(float(total_score))

            sub = item_report.get("sub_scores", {})
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if isinstance(v, (int, float)):
                        sub_acc.setdefault(str(k), []).append(float(v))

        out: dict[str, float] = {}
        if totals:
            out["proxy_total_score"] = float(sum(totals) / len(totals))
        for k, arr in sub_acc.items():
            out[f"proxy_{k}"] = float(sum(arr) / len(arr))
        return out

    out = {}
    total = report.get("total_score")
    if isinstance(total, (int, float)):
        out["proxy_total_score"] = float(total)

    sub = report.get("sub_scores", {})
    if isinstance(sub, dict):
        for k, v in sub.items():
            if isinstance(v, (int, float)):
                out[f"proxy_{k}"] = float(v)

    flags = report.get("flags", {})
    if isinstance(flags, dict):
        hf = flags.get("hard_fail")
        if isinstance(hf, bool):
            out["proxy_hard_fail"] = 1.0 if hf else 0.0

    return out