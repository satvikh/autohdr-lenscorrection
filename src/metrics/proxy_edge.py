"""Edge-based proxy score component."""

from __future__ import annotations

import numpy as np

from src.metrics._image_utils import to_rgb_float_array
from src.metrics._utils import _safe_float, _to_gray


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude with vectorized NumPy ops."""
    p = np.pad(gray.astype(np.float32, copy=False), ((1, 1), (1, 1)), mode="edge")

    gx = (
        -p[:-2, :-2]
        + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2]
        + 2.0 * p[1:-1, 2:]
        - p[2:, :-2]
        + p[2:, 2:]
    )
    gy = (
        p[:-2, :-2]
        + 2.0 * p[:-2, 1:-1]
        + p[:-2, 2:]
        - p[2:, :-2]
        - 2.0 * p[2:, 1:-1]
        - p[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _normalize_map(edge_mag: np.ndarray, eps: float) -> np.ndarray:
    max_v = float(np.max(edge_mag)) if edge_mag.size else 0.0
    return edge_mag / (max_v + eps)


def compute_edge_score(pred, gt, config) -> float:
    """Compute Sobel-edge similarity score in [0, 1]."""
    cfg = config or {}
    eps = float(cfg.get("edge_eps", 1e-8))
    similarity = str(cfg.get("edge_similarity", "cosine")).lower()

    pred_np = to_rgb_float_array(pred)
    gt_np = to_rgb_float_array(gt)
    if pred_np.shape != gt_np.shape:
        raise ValueError(f"Shape mismatch: pred={pred_np.shape}, gt={gt_np.shape}")

    pred_mag = _normalize_map(_sobel_magnitude(_to_gray(pred_np)), eps)
    gt_mag = _normalize_map(_sobel_magnitude(_to_gray(gt_np)), eps)

    pred_flat = pred_mag.reshape(-1).astype(np.float64, copy=False)
    gt_flat = gt_mag.reshape(-1).astype(np.float64, copy=False)

    if similarity == "mae":
        score = 1.0 - float(np.mean(np.abs(pred_flat - gt_flat)))
        return _safe_float(score)

    # Default: cosine similarity over normalized edge maps.
    pred_norm = float(np.linalg.norm(pred_flat))
    gt_norm = float(np.linalg.norm(gt_flat))
    denom = pred_norm * gt_norm
    if denom <= eps:
        if pred_norm <= eps and gt_norm <= eps:
            return 1.0
        return 0.0
    score = float(np.dot(pred_flat, gt_flat) / denom)
    return _safe_float(score)
