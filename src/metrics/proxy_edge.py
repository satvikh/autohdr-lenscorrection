"""Edge proxy score using multi-scale Canny-style F1 similarity."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.metrics._image_utils import to_rgb_float_array
from src.metrics._utils import _safe_float, _to_gray


def _resize_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    if scale <= 0.0:
        raise ValueError(f"edge scale must be > 0, got {scale}")

    h, w = gray.shape
    oh = max(int(round(h * scale)), 4)
    ow = max(int(round(w * scale)), 4)
    try:
        import cv2  # type: ignore

        out = cv2.resize(gray.astype(np.float32, copy=False), (ow, oh), interpolation=cv2.INTER_AREA)
        return out.astype(np.float32, copy=False)
    except Exception:
        # Nearest fallback for environments without cv2.
        y_idx = np.linspace(0, h - 1, num=oh).round().astype(np.int32)
        x_idx = np.linspace(0, w - 1, num=ow).round().astype(np.int32)
        return gray[np.ix_(y_idx, x_idx)].astype(np.float32, copy=False)


def _canny_edges(gray: np.ndarray, low: float, high: float) -> np.ndarray:
    gray_safe = np.nan_to_num(gray.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    g = np.clip(gray_safe * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
    lo = int(np.clip(round(low * 255.0), 0, 255))
    hi = int(np.clip(round(high * 255.0), 0, 255))
    if hi < lo:
        lo, hi = hi, lo

    try:
        import cv2  # type: ignore

        e = cv2.Canny(g, threshold1=lo, threshold2=hi)
        return (e > 0).astype(np.uint8)
    except Exception:
        # Lightweight gradient-threshold fallback.
        gy, gx = np.gradient(gray.astype(np.float32, copy=False))
        mag = np.sqrt(gx * gx + gy * gy)
        thr = float(np.quantile(mag.reshape(-1), 0.80)) if mag.size else 0.0
        return (mag >= thr).astype(np.uint8)


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    try:
        import cv2  # type: ignore

        k = int(2 * radius + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        return (cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1) > 0).astype(np.uint8)
    except Exception:
        m = (mask > 0).astype(np.uint8)
        out = m.copy()
        h, w = m.shape
        padded = np.pad(m, ((radius, radius), (radius, radius)), mode="constant", constant_values=0)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ys = radius + dy
                xs = radius + dx
                shifted = padded[ys : ys + h, xs : xs + w]
                out = np.maximum(out, shifted)
        return out


def _edge_f1(pred_edges: np.ndarray, gt_edges: np.ndarray, tolerance_px: int, eps: float) -> float:
    pred = (pred_edges > 0).astype(np.uint8)
    gt = (gt_edges > 0).astype(np.uint8)

    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    pred_d = _dilate_binary(pred, radius=tolerance_px)
    gt_d = _dilate_binary(gt, radius=tolerance_px)

    tp_pred = float((pred & gt_d).sum())
    tp_gt = float((gt & pred_d).sum())

    precision = tp_pred / max(float(pred_count), eps)
    recall = tp_gt / max(float(gt_count), eps)
    denom = precision + recall
    if denom <= eps:
        return 0.0
    return float((2.0 * precision * recall) / denom)


def _normalize_scale_weights(scales: Sequence[float], weights: Sequence[float] | None) -> np.ndarray:
    n = len(scales)
    if n == 0:
        raise ValueError("edge_scales must be non-empty")
    if weights is None:
        out = np.ones((n,), dtype=np.float64)
    else:
        if len(weights) != n:
            raise ValueError("edge_scale_weights must match edge_scales length")
        out = np.asarray(weights, dtype=np.float64)
    s = float(out.sum())
    if s <= 0.0:
        out = np.ones((n,), dtype=np.float64) / float(n)
    else:
        out = out / s
    return out


def compute_edge_score(pred, gt, config) -> float:
    """Compute edge similarity in [0,1] using multi-scale Canny-style F1."""
    cfg = config or {}
    eps = float(cfg.get("edge_eps", 1e-8))
    canny_low = float(cfg.get("edge_canny_low", 0.08))
    canny_high = float(cfg.get("edge_canny_high", 0.20))
    tolerance_px = max(int(cfg.get("edge_tolerance_px", 1)), 0)
    scales = [float(s) for s in cfg.get("edge_scales", [1.0, 0.5])]
    scale_weights = cfg.get("edge_scale_weights")
    if isinstance(scale_weights, (list, tuple)):
        scale_weights = [float(x) for x in scale_weights]
    else:
        scale_weights = None
    w = _normalize_scale_weights(scales, scale_weights)

    pred_np = to_rgb_float_array(pred)
    gt_np = to_rgb_float_array(gt)
    if pred_np.shape != gt_np.shape:
        raise ValueError(f"Shape mismatch: pred={pred_np.shape}, gt={gt_np.shape}")

    pred_gray = _to_gray(pred_np).astype(np.float32, copy=False)
    gt_gray = _to_gray(gt_np).astype(np.float32, copy=False)

    score = 0.0
    for i, scale in enumerate(scales):
        p = _resize_gray(pred_gray, scale)
        g = _resize_gray(gt_gray, scale)
        p_edges = _canny_edges(p, canny_low, canny_high)
        g_edges = _canny_edges(g, canny_low, canny_high)
        score += float(w[i]) * _edge_f1(p_edges, g_edges, tolerance_px=tolerance_px, eps=eps)
    return _safe_float(score)
