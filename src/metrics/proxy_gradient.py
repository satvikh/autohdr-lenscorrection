"""Gradient-orientation proxy placeholder module."""

from __future__ import annotations

from src.metrics.proxy_edge import compute_edge_score


def compute_gradient_score(pred, gt, config) -> float:
    """Compute gradient consistency score in [0, 1] (placeholder via edge similarity)."""
    return compute_edge_score(pred, gt, config)
