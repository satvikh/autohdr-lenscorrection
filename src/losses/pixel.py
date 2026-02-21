from __future__ import annotations

import torch
from torch import Tensor
from torch import nn


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Elementwise L1 loss with strict shape checking.

    Args:
        pred: Tensor[B, C, H, W]
        target: Tensor[B, C, H, W]

    Returns:
        Scalar tensor (mean absolute error).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shape, got {pred.shape} vs {target.shape}")
    return (pred - target).abs().mean()


class CharbonnierLoss(nn.Module):
    """Differentiable robust pixel loss.

    Computes mean(sqrt((pred-target)^2 + eps^2)).

    Shapes:
        pred: Tensor[B, C, H, W]
        target: Tensor[B, C, H, W]
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have identical shape, got {pred.shape} vs {target.shape}")
        diff = pred - target
        return torch.sqrt((diff * diff) + (self.eps * self.eps)).mean()