"""Line-consistency proxy score using orientation-distribution matching."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.metrics._image_utils import to_rgb_float_array
from src.metrics._utils import _safe_float, _to_gray
from src.metrics.proxy_edge import _canny_edges, _normalize_scale_weights, _resize_gray


def _line_segments_hough(edges: np.ndarray, *, min_len: float, threshold: int, max_gap: float) -> list[tuple[float, float]]:
    """Return list of (orientation_rad in [0,pi), length)."""
    out: list[tuple[float, float]] = []
    try:
        import cv2  # type: ignore

        lines = cv2.HoughLinesP(
            (edges > 0).astype(np.uint8) * 255,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=int(max(threshold, 1)),
            minLineLength=float(max(min_len, 1.0)),
            maxLineGap=float(max(max_gap, 0.0)),
        )
        if lines is None:
            return out
        for line in lines:
            x1, y1, x2, y2 = [float(v) for v in line[0]]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length <= 0.0:
                continue
            theta = float(np.arctan2(dy, dx))
            theta = float(np.mod(theta, np.pi))
            out.append((theta, length))
        return out
    except Exception:
        return out


def _line_segments_lsd(gray: np.ndarray, *, min_len: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    try:
        import cv2  # type: ignore

        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = lsd.detect(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8))
        lines = detected[0] if isinstance(detected, tuple) else detected
        if lines is None:
            return out
        for line in lines:
            x1, y1, x2, y2 = [float(v) for v in line[0]]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_len:
                continue
            theta = float(np.mod(np.arctan2(dy, dx), np.pi))
            out.append((theta, length))
        return out
    except Exception:
        return out


def _fallback_orientation_segments(gray: np.ndarray, *, canny_low: float, canny_high: float) -> list[tuple[float, float]]:
    """Fallback orientation samples when cv2 line detectors are unavailable.

    Treats strong edge pixels as short line elements, using gradient orientation
    rotated by 90 degrees (line direction is perpendicular to edge gradient).
    """
    edges = _canny_edges(gray, canny_low, canny_high).astype(bool)
    if not np.any(edges):
        return []
    gy, gx = np.gradient(gray.astype(np.float32, copy=False))
    mag = np.sqrt((gx * gx) + (gy * gy)).astype(np.float32, copy=False)
    theta_line = np.mod(np.arctan2(gy, gx) + (0.5 * np.pi), np.pi).astype(np.float32, copy=False)
    idx = np.where(edges)
    if idx[0].size == 0:
        return []
    thetas = theta_line[idx]
    weights = np.maximum(mag[idx], 1e-6)
    return [(float(t), float(w)) for t, w in zip(thetas.tolist(), weights.tolist())]


def _orientation_hist(
    segments: Sequence[tuple[float, float]],
    *,
    bins: int,
    smooth_sigma_bins: float,
    eps: float,
) -> np.ndarray:
    hist = np.zeros((bins,), dtype=np.float64)
    if not segments:
        return hist

    period = np.pi
    centers = (np.arange(bins, dtype=np.float64) + 0.5) * (period / float(bins))

    for theta, weight in segments:
        if weight <= 0.0:
            continue
        diff = np.abs(theta - centers)
        diff = np.minimum(diff, period - diff)
        if smooth_sigma_bins > 0.0:
            sigma = (period / float(bins)) * smooth_sigma_bins
            contrib = np.exp(-0.5 * (diff / max(sigma, eps)) ** 2)
        else:
            idx = int(np.floor(theta / period * bins)) % bins
            contrib = np.zeros((bins,), dtype=np.float64)
            contrib[idx] = 1.0
        hist += float(weight) * contrib

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


def _segments_for_gray(
    gray: np.ndarray,
    *,
    canny_low: float,
    canny_high: float,
    min_len: float,
    hough_threshold: int,
    hough_max_gap: float,
    use_lsd: bool,
) -> list[tuple[float, float]]:
    edges = _canny_edges(gray, canny_low, canny_high)
    segs = _line_segments_hough(edges, min_len=min_len, threshold=hough_threshold, max_gap=hough_max_gap)
    if use_lsd:
        segs.extend(_line_segments_lsd(gray, min_len=min_len))
    if not segs:
        segs = _fallback_orientation_segments(gray, canny_low=canny_low, canny_high=canny_high)
    return segs


def compute_line_score(pred, gt, config) -> float:
    """Compute line-consistency score in [0,1] via orientation distributions."""
    cfg = config or {}
    eps = float(cfg.get("line_eps", 1e-8))
    bins = max(int(cfg.get("line_hist_bins", 36)), 8)
    min_len = float(cfg.get("line_min_length_px", 12.0))
    hough_threshold = int(cfg.get("line_hough_threshold", 24))
    hough_max_gap = float(cfg.get("line_hough_max_gap_px", 6.0))
    canny_low = float(cfg.get("line_canny_low", cfg.get("edge_canny_low", 0.08)))
    canny_high = float(cfg.get("line_canny_high", cfg.get("edge_canny_high", 0.20)))
    smooth_sigma_bins = float(cfg.get("line_hist_smooth_sigma_bins", 1.25))
    use_lsd = bool(cfg.get("line_use_lsd", False))
    scales = [float(s) for s in cfg.get("line_scales", [1.0, 0.5])]
    scale_weights = cfg.get("line_scale_weights")
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

    total = 0.0
    for i, scale in enumerate(scales):
        p = _resize_gray(pred_gray, scale)
        g = _resize_gray(gt_gray, scale)
        seg_p = _segments_for_gray(
            p,
            canny_low=canny_low,
            canny_high=canny_high,
            min_len=max(min_len * scale, 4.0),
            hough_threshold=hough_threshold,
            hough_max_gap=hough_max_gap * max(scale, eps),
            use_lsd=use_lsd,
        )
        seg_g = _segments_for_gray(
            g,
            canny_low=canny_low,
            canny_high=canny_high,
            min_len=max(min_len * scale, 4.0),
            hough_threshold=hough_threshold,
            hough_max_gap=hough_max_gap * max(scale, eps),
            use_lsd=use_lsd,
        )
        hist_p = _orientation_hist(seg_p, bins=bins, smooth_sigma_bins=smooth_sigma_bins, eps=eps)
        hist_g = _orientation_hist(seg_g, bins=bins, smooth_sigma_bins=smooth_sigma_bins, eps=eps)
        total += float(w[i]) * _hist_similarity(hist_p, hist_g, eps=eps)
    return _safe_float(total)
