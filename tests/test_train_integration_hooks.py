from __future__ import annotations

import torch

from src.metrics.proxy_score import compute_proxy_score
from src.train.proxy_hooks import compute_proxy_metrics_for_batch, resolve_proxy_scorer


def test_resolve_proxy_scorer_missing_module_returns_none() -> None:
    scorer = resolve_proxy_scorer(module_path="nonexistent.module", function_name="compute_proxy_score")
    assert scorer is None


def test_proxy_metrics_batch_path() -> None:
    def scorer(pred: torch.Tensor, gt: torch.Tensor, config=None):
        return {
            "total_score": 0.42,
            "sub_scores": {"edge": 0.5, "line": 0.4, "grad": 0.3, "ssim": 0.2, "mae": 0.1},
            "flags": {"hard_fail": False, "reasons": []},
        }

    pred = torch.rand(2, 3, 16, 16)
    gt = torch.rand(2, 3, 16, 16)

    out = compute_proxy_metrics_for_batch(scorer=scorer, pred_batch=pred, target_batch=gt, config=None)
    assert abs(out["proxy_total_score"] - 0.42) < 1e-8
    assert "proxy_edge" in out
    assert out["proxy_hard_fail"] == 0.0


def test_proxy_metrics_per_sample_fallback() -> None:
    def scorer(pred: torch.Tensor, gt: torch.Tensor, config=None):
        if pred.shape[0] > 1:
            raise RuntimeError("batch mode unsupported")
        return {
            "total_score": float(pred.mean().item()),
            "sub_scores": {"edge": 0.5},
            "flags": {"hard_fail": False, "reasons": []},
        }

    pred = torch.rand(3, 3, 8, 8)
    gt = torch.rand(3, 3, 8, 8)

    out = compute_proxy_metrics_for_batch(scorer=scorer, pred_batch=pred, target_batch=gt, config=None)
    assert "proxy_total_score" in out
    assert "proxy_edge" in out


def test_proxy_metrics_with_builtin_scorer_handles_batched_tensors() -> None:
    pred = torch.rand(2, 3, 16, 16)
    gt = pred.clone()

    out = compute_proxy_metrics_for_batch(scorer=compute_proxy_score, pred_batch=pred, target_batch=gt, config={})

    assert "proxy_total_score" in out
    assert out["proxy_total_score"] > 0.9
    assert out.get("proxy_hard_fail", 1.0) == 0.0


def test_proxy_metrics_with_builtin_scorer_handles_batch_size_one() -> None:
    pred = torch.rand(1, 3, 16, 16)
    gt = pred.clone()

    out = compute_proxy_metrics_for_batch(scorer=compute_proxy_score, pred_batch=pred, target_batch=gt, config={})

    assert "proxy_total_score" in out
    assert out["proxy_total_score"] > 0.9
    assert out.get("proxy_hard_fail", 1.0) == 0.0


def test_proxy_metrics_hardfail_nonfinite_totals_penalize_to_zero() -> None:
    def scorer(pred: torch.Tensor, gt: torch.Tensor, config=None):
        _ = pred
        _ = gt
        _ = config
        return {
            "total_score": float("nan"),
            "sub_scores": {"edge": float("nan")},
            "flags": {"hard_fail": True, "reasons": ["synthetic_fail"]},
        }

    pred = torch.rand(2, 3, 16, 16)
    gt = torch.rand(2, 3, 16, 16)
    out = compute_proxy_metrics_for_batch(
        scorer=scorer,
        pred_batch=pred,
        target_batch=gt,
        config={"hardfail": {"penalty_mode": "score_zero"}},
    )
    assert out["proxy_total_score"] == 0.0
    assert out.get("proxy_hard_fail", 0.0) == 1.0


def test_proxy_metrics_hardfail_nonfinite_totals_default_to_clamp_penalty() -> None:
    def scorer(pred: torch.Tensor, gt: torch.Tensor, config=None):
        _ = pred
        _ = gt
        _ = config
        return {
            "total_score": float("nan"),
            "sub_scores": {"edge": float("nan")},
            "flags": {"hard_fail": True, "reasons": ["synthetic_fail"]},
        }

    pred = torch.rand(2, 3, 16, 16)
    gt = torch.rand(2, 3, 16, 16)
    out = compute_proxy_metrics_for_batch(scorer=scorer, pred_batch=pred, target_batch=gt, config={})
    assert abs(float(out["proxy_total_score"]) - 0.05) < 1e-8
    assert out.get("proxy_hard_fail", 0.0) == 1.0
