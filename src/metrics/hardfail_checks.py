"""Hard-fail screening for catastrophic outputs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.metrics._image_utils import to_rgb_float_array
from src.metrics._utils import _to_gray
from src.metrics.proxy_ssim_mae import compute_mae


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

    mae_max = float(hcfg.get("mae_max", 0.75))
    mae = compute_mae(pred_np, gt_np, cfg)
    if mae > mae_max:
        reasons.append("mae_too_high")

    gt_std_min = float(hcfg.get("gt_std_min", 0.02))
    pred_std_max = float(hcfg.get("pred_std_max", 0.002))
    if float(_to_gray(gt_np).std()) > gt_std_min and float(_to_gray(pred_np).std()) < pred_std_max:
        reasons.append("edge_collapse")

    return len(reasons) > 0, reasons
