from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def ssim_index(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    window_size: int = 11,
    eps: float = 1e-8,
) -> Tensor:
    """Compute SSIM index map mean over NCHW images.

    Shapes:
        pred: Tensor[B, C, H, W]
        target: Tensor[B, C, H, W]
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shape, got {pred.shape} vs {target.shape}")
    if pred.ndim != 4:
        raise ValueError("pred and target must have shape [B,C,H,W]")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(pred * pred, kernel_size=window_size, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=window_size, stride=1, padding=pad) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_map = num / (den + eps)
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    """Loss wrapper: 1 - SSIM."""

    def __init__(
        self,
        *,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        window_size: int = 11,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.data_range = float(data_range)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.window_size = int(window_size)
        self.eps = float(eps)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        score = ssim_index(
            pred,
            target,
            data_range=self.data_range,
            k1=self.k1,
            k2=self.k2,
            window_size=self.window_size,
            eps=self.eps,
        )
        return torch.clamp(1.0 - score, min=0.0)