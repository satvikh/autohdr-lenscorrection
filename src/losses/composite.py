from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from torch import nn

from src.losses.flow_regularizers import flow_curvature_loss, flow_magnitude_loss, total_variation_loss
from src.losses.gradients import edge_magnitude_loss, gradient_orientation_cosine_loss, multiscale_average
from src.losses.jacobian_loss import jacobian_foldover_penalty
from src.losses.pixel import CharbonnierLoss, l1_loss
from src.losses.ssim_loss import SSIMLoss


@dataclass(frozen=True)
class CompositeLossConfig:
    stage: str = "stage1_param_only"
    use_charbonnier: bool = True
    charbonnier_eps: float = 1e-3
    multiscale_scales: tuple[float, ...] = (1.0, 0.5)

    pixel_weight: float = 0.10
    ssim_weight: float = 0.15
    edge_weight: float = 0.40
    grad_orient_weight: float = 0.18

    flow_tv_weight: float = 0.0
    flow_mag_weight: float = 0.0
    flow_curv_weight: float = 0.0
    jacobian_weight: float = 0.0

    jacobian_margin: float = 0.0


class CompositeLoss(nn.Module):
    """Composite training loss with stage-aware regularization.

    Inputs:
        pred_image: Tensor[B, C, H, W]
        target_image: Tensor[B, C, H, W]
        residual_flow_lowres: optional Tensor[B, 2, Hr, Wr]
        final_grid_bhwc: optional Tensor[B, H, W, 2]

    Returns:
        total_loss: scalar Tensor
        components: dict[str, Tensor]
    """

    def __init__(self, config: CompositeLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or CompositeLossConfig()

        self.pixel_charb = CharbonnierLoss(eps=self.config.charbonnier_eps)
        self.ssim_loss = SSIMLoss()

    @staticmethod
    def _zeros_like_ref(ref: Tensor) -> Tensor:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    def _image_pixel_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.config.use_charbonnier:
            return self.pixel_charb(pred, target)
        return l1_loss(pred, target)

    def _multiscale(self, pred: Tensor, target: Tensor, fn: Any) -> Tensor:
        scales: Sequence[float] = self.config.multiscale_scales
        return multiscale_average(pred, target, scales=scales, loss_fn=fn)

    def forward(
        self,
        pred_image: Tensor,
        target_image: Tensor,
        *,
        residual_flow_lowres: Tensor | None = None,
        final_grid_bhwc: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if pred_image.shape != target_image.shape:
            raise ValueError(
                f"pred_image and target_image must have identical shape, got {pred_image.shape} vs {target_image.shape}"
            )

        pix = self._multiscale(pred_image, target_image, self._image_pixel_loss)
        ssim = self._multiscale(pred_image, target_image, self.ssim_loss)
        edge = self._multiscale(pred_image, target_image, edge_magnitude_loss)
        grad = self._multiscale(pred_image, target_image, gradient_orientation_cosine_loss)

        flow_tv = self._zeros_like_ref(pred_image)
        flow_mag = self._zeros_like_ref(pred_image)
        flow_curv = self._zeros_like_ref(pred_image)
        if residual_flow_lowres is not None:
            flow_tv = total_variation_loss(residual_flow_lowres)
            flow_mag = flow_magnitude_loss(residual_flow_lowres)
            flow_curv = flow_curvature_loss(residual_flow_lowres)

        jac = self._zeros_like_ref(pred_image)
        if final_grid_bhwc is not None:
            jac = jacobian_foldover_penalty(final_grid_bhwc, margin=self.config.jacobian_margin)

        components = {
            "pixel": pix,
            "ssim": ssim,
            "edge": edge,
            "grad_orient": grad,
            "flow_tv": flow_tv,
            "flow_mag": flow_mag,
            "flow_curv": flow_curv,
            "jacobian": jac,
            "pixel_weighted": self.config.pixel_weight * pix,
            "ssim_weighted": self.config.ssim_weight * ssim,
            "edge_weighted": self.config.edge_weight * edge,
            "grad_orient_weighted": self.config.grad_orient_weight * grad,
            "flow_tv_weighted": self.config.flow_tv_weight * flow_tv,
            "flow_mag_weighted": self.config.flow_mag_weight * flow_mag,
            "flow_curv_weighted": self.config.flow_curv_weight * flow_curv,
            "jacobian_weighted": self.config.jacobian_weight * jac,
        }

        total = (
            components["pixel_weighted"]
            + components["ssim_weighted"]
            + components["edge_weighted"]
            + components["grad_orient_weighted"]
            + components["flow_tv_weighted"]
            + components["flow_mag_weighted"]
            + components["flow_curv_weighted"]
            + components["jacobian_weighted"]
        )

        components["total"] = total
        return total, components


def config_for_stage(stage: str) -> CompositeLossConfig:
    s = stage.lower()
    if s == "stage1_param_only":
        return CompositeLossConfig(
            stage=s,
            flow_tv_weight=0.0,
            flow_mag_weight=0.0,
            flow_curv_weight=0.0,
            jacobian_weight=0.0,
        )
    if s == "stage2_hybrid":
        return CompositeLossConfig(
            stage=s,
            flow_tv_weight=0.05,
            flow_mag_weight=0.02,
            flow_curv_weight=0.01,
            jacobian_weight=0.01,
        )
    if s == "stage3_finetune":
        return CompositeLossConfig(
            stage=s,
            pixel_weight=0.08,
            edge_weight=0.45,
            grad_orient_weight=0.20,
            flow_tv_weight=0.05,
            flow_mag_weight=0.02,
            flow_curv_weight=0.01,
            jacobian_weight=0.01,
        )
    raise ValueError(f"Unsupported stage: {stage}")
