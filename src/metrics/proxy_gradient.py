"""Gradient-orientation proxy score via weighted orientation histograms."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.metrics._image_utils import to_rgb_float_array
from src.metrics._utils import _safe_float, _to_gray
from src.metrics.proxy_edge import _normalize_scale_weights, _resize_gray


def _sobel_xy(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = gray.astype(np.float32, copy=False)
    try:
        import cv2  # type: ignore

        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
        return gx, gy
    except Exception:
        p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
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
        return gx.astype(np.float32, copy=False), gy.astype(np.float32, copy=False)


def _orientation_hist(
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    bins: int,
    eps: float,
    mag_power: float,
    mag_threshold_quantile: float,
) -> np.ndarray:
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float64, copy=False)
    theta = np.mod(np.arctan2(gy, gx), np.pi).astype(np.float64, copy=False)

    if mag.size == 0:
        return np.zeros((bins,), dtype=np.float64)

    q = float(np.clip(mag_threshold_quantile, 0.0, 1.0))
    thr = float(np.quantile(mag.reshape(-1), q)) if q > 0.0 else 0.0
    weights = np.where(mag >= thr, np.power(np.maximum(mag, 0.0), mag_power), 0.0)

    hist = np.zeros((bins,), dtype=np.float64)
    idx = np.floor(theta / np.pi * bins).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    np.add.at(hist, idx.reshape(-1), weights.reshape(-1))

    total = float(hist.sum())
    if total <= eps:
        return np.zeros((bins,), dtype=np.float64)
    return hist / total


def _hist_similarity(pred_hist: np.ndarray, gt_hist: np.ndarray, eps: float) -> float:
    p = pred_hist.astype(np.float64, copy=False)
    g = gt_hist.astype(np.float64, copy=False)
    pn = float(np.linalg.norm(p))
    gn = float(np.linalg.norm(g))
    if pn <= eps and gn <= eps:
        return 1.0
    if pn <= eps or gn <= eps:
        return 0.0
    cosine = float(np.dot(p, g) / max(pn * gn, eps))
    bc = float(np.sum(np.sqrt(np.clip(p, 0.0, 1.0) * np.clip(g, 0.0, 1.0))))
    return float(np.clip(0.5 * (cosine + bc), 0.0, 1.0))


def compute_gradient_score(pred, gt, config) -> float:
    """Compute gradient-orientation similarity in [0,1]."""
    cfg = config or {}
    eps = float(cfg.get("grad_eps", 1e-8))
    bins = max(int(cfg.get("grad_hist_bins", 36)), 8)
    mag_power = float(cfg.get("grad_mag_power", 1.0))
    mag_thr_q = float(cfg.get("grad_mag_threshold_quantile", 0.10))
    scales = [float(s) for s in cfg.get("grad_scales", [1.0, 0.5])]
    scale_weights = cfg.get("grad_scale_weights")
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
        pgx, pgy = _sobel_xy(p)
        ggx, ggy = _sobel_xy(g)
        hist_p = _orientation_hist(
            pgx,
            pgy,
            bins=bins,
            eps=eps,
            mag_power=mag_power,
            mag_threshold_quantile=mag_thr_q,
        )
        hist_g = _orientation_hist(
            ggx,
            ggy,
            bins=bins,
            eps=eps,
            mag_power=mag_power,
            mag_threshold_quantile=mag_thr_q,
        )
        score += float(w[i]) * _hist_similarity(hist_p, hist_g, eps=eps)
    return _safe_float(score)
