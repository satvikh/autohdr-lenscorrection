"""SSIM and MAE proxy component implementations."""

from __future__ import annotations

import numpy as np

from src.metrics._image_utils import to_rgb_float_array
from src.metrics._utils import _safe_float, _to_gray


def _global_ssim_luma(pred_rgb: np.ndarray, gt_rgb: np.ndarray, c1: float, c2: float) -> float:
    pred_luma = _to_gray(pred_rgb)
    gt_luma = _to_gray(gt_rgb)

    mu_x = float(pred_luma.mean())
    mu_y = float(gt_luma.mean())
    sigma_x = float(pred_luma.var())
    sigma_y = float(gt_luma.var())
    sigma_xy = float(((pred_luma - mu_x) * (gt_luma - mu_y)).mean())

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    if den == 0.0:
        return 1.0
    return _safe_float(num / den)


def _local_ssim_luma(pred_rgb: np.ndarray, gt_rgb: np.ndarray, *, c1: float, c2: float) -> float:
    """Compute windowed SSIM using Gaussian local statistics (skimage-like fallback)."""
    x = _to_gray(pred_rgb).astype(np.float32, copy=False)
    y = _to_gray(gt_rgb).astype(np.float32, copy=False)
    try:
        import cv2  # type: ignore

        mu_x = cv2.GaussianBlur(x, ksize=(11, 11), sigmaX=1.5, borderType=cv2.BORDER_REPLICATE)
        mu_y = cv2.GaussianBlur(y, ksize=(11, 11), sigmaX=1.5, borderType=cv2.BORDER_REPLICATE)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(x * x, ksize=(11, 11), sigmaX=1.5, borderType=cv2.BORDER_REPLICATE) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, ksize=(11, 11), sigmaX=1.5, borderType=cv2.BORDER_REPLICATE) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, ksize=(11, 11), sigmaX=1.5, borderType=cv2.BORDER_REPLICATE) - mu_xy

        num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        den = np.maximum(den, 1e-12)
        ssim_map = num / den
        return _safe_float(float(np.mean(ssim_map)))
    except Exception:
        return _global_ssim_luma(pred_rgb, gt_rgb, c1=c1, c2=c2)


def compute_ssim(pred, gt, config=None) -> float:
    """Compute SSIM in [0, 1], using skimage when available with global fallback."""
    cfg = config or {}
    c1 = float(cfg.get("ssim_c1", (0.01 ** 2)))
    c2 = float(cfg.get("ssim_c2", (0.03 ** 2)))

    pred_np = to_rgb_float_array(pred)
    gt_np = to_rgb_float_array(gt)
    if pred_np.shape != gt_np.shape:
        raise ValueError(f"Shape mismatch: pred={pred_np.shape}, gt={gt_np.shape}")

    try:
        from skimage.metrics import structural_similarity as ssim_fn  # type: ignore

        pred_luma = _to_gray(pred_np)
        gt_luma = _to_gray(gt_np)
        try:
            score = float(ssim_fn(pred_luma, gt_luma, data_range=1.0))
        except TypeError:
            score = float(ssim_fn(pred_luma, gt_luma))
        return _safe_float(score)
    except Exception:
        return _local_ssim_luma(pred_np, gt_np, c1=c1, c2=c2)


def compute_mae(pred, gt, config=None) -> float:
    """Compute mean absolute error over [0,1] pixels in [0, 1]."""
    pred_np = to_rgb_float_array(pred)
    gt_np = to_rgb_float_array(gt)
    if pred_np.shape != gt_np.shape:
        raise ValueError(f"Shape mismatch: pred={pred_np.shape}, gt={gt_np.shape}")
    mae = float(np.mean(np.abs(pred_np - gt_np)))
    return _safe_float(mae)
