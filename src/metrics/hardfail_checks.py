"""Hard-fail screening for catastrophic outputs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.metrics._image_utils import to_rgb_float_array
from src.metrics._utils import _to_gray
from src.metrics.proxy_edge import compute_edge_score
from src.metrics.proxy_gradient import compute_gradient_score
from src.metrics.proxy_ssim_mae import compute_mae


def _regional_max_diff(pred_np: np.ndarray, gt_np: np.ndarray, *, window: int) -> float:
    """Compute max local mean absolute difference over RGB channels in [0,1]."""
    if pred_np.shape != gt_np.shape:
        return float("inf")
    if window <= 1:
        return float(np.max(np.mean(np.abs(pred_np - gt_np), axis=2)))

    diff = np.mean(np.abs(pred_np - gt_np), axis=2).astype(np.float32, copy=False)
    k = int(max(window, 1))
    try:
        import cv2  # type: ignore

        local = cv2.blur(diff, ksize=(k, k), borderType=cv2.BORDER_REPLICATE)
        return float(np.max(local)) if local.size else 0.0
    except Exception:
        # Integral-image fallback.
        p = np.pad(diff, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        integral = np.cumsum(np.cumsum(p, axis=0), axis=1)
        h, w = diff.shape
        hh = min(k, h)
        ww = min(k, w)
        if hh <= 0 or ww <= 0:
            return 0.0
        max_mean = 0.0
        for y in range(0, h - hh + 1):
            y2 = y + hh
            for x in range(0, w - ww + 1):
                x2 = x + ww
                total = integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
                mean = float(total) / float(hh * ww)
                if mean > max_mean:
                    max_mean = mean
        return max_mean


def run_hardfail_checks(pred, gt, config) -> Tuple[bool, List[str]]:
    """Return hard-fail flag with reasons for critical quality/integrity failures."""
    cfg = config or {}
    hcfg = cfg.get("hardfail", {}) if isinstance(cfg, dict) else {}
    image_cfg = cfg.get("image", {}) if isinstance(cfg, dict) else {}
    require_same_size = bool(image_cfg.get("require_same_size", False))

    reasons: List[str] = []
    try:
        pred_np = to_rgb_float_array(pred)
    except ValueError as exc:
        reasons.append(f"pred_load_error:{exc}")
        pred_np = None
    try:
        gt_np = to_rgb_float_array(gt)
    except ValueError as exc:
        reasons.append(f"gt_load_error:{exc}")
        gt_np = None

    if pred_np is None or gt_np is None:
        return True, reasons

    if require_same_size and pred_np.shape[:2] != gt_np.shape[:2]:
        reasons.append(f"size_mismatch: pred_hw={pred_np.shape[:2]}, gt_hw={gt_np.shape[:2]}")
    if pred_np.shape != gt_np.shape:
        reasons.append("shape_mismatch")
    if not np.isfinite(pred_np).all() or not np.isfinite(gt_np).all():
        reasons.append("non_finite_values")

    if pred_np.size == 0 or gt_np.size == 0:
        reasons.append("empty_image")

    regional_window = int(hcfg.get("regional_window", 64))
    regional_max_allowed = float(hcfg.get("regional_max", 0.30))
    regional_max = _regional_max_diff(pred_np, gt_np, window=regional_window)
    if regional_max > regional_max_allowed:
        reasons.append("regional_max_diff_too_high")

    edge_min = float(hcfg.get("edge_min", 0.05))
    try:
        edge_score = float(compute_edge_score(pred_np, gt_np, cfg))
        if edge_score < edge_min:
            reasons.append("edge_similarity_below_min")
    except Exception as exc:
        reasons.append(f"edge_score_error:{exc}")

    grad_min = hcfg.get("grad_min", None)
    if grad_min is not None:
        try:
            grad_score = float(compute_gradient_score(pred_np, gt_np, cfg))
            if grad_score < float(grad_min):
                reasons.append("gradient_similarity_below_min")
        except Exception as exc:
            reasons.append(f"gradient_score_error:{exc}")

    mae_max_raw = hcfg.get("mae_max", None)
    if mae_max_raw is not None:
        mae_max = float(mae_max_raw)
        mae = compute_mae(pred_np, gt_np, cfg)
        if mae > mae_max:
            reasons.append("mae_too_high")

    edge_collapse_enabled = bool(hcfg.get("edge_collapse_enabled", False))
    if edge_collapse_enabled:
        gt_std_min = float(hcfg.get("gt_std_min", 0.02))
        pred_std_max = float(hcfg.get("pred_std_max", 0.002))
        if float(_to_gray(gt_np).std()) > gt_std_min and float(_to_gray(pred_np).std()) < pred_std_max:
            reasons.append("edge_collapse")

    return len(reasons) > 0, reasons
