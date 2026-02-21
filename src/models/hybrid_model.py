from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from src.models.backbones import create_backbone
from src.models.coord_channels import CoordChannelAppender
from src.models.heads_parametric import ParametricBounds, ParametricHead
from src.models.heads_residual import ResidualFlowHead


class ModelWarpBackend(Protocol):
    """Optional model-forward warp backend protocol.

    Inputs:
        image: Tensor[B, 3, H, W]
        params: Tensor[B, N]
        residual_flow_lowres: Tensor[B, 2, Hr, Wr] or None

    Returns:
        Dict containing at least `pred_image` when used.
    """

    def __call__(
        self,
        image: Tensor,
        params: Tensor,
        residual_flow_lowres: Tensor | None,
    ) -> dict[str, Tensor]:
        ...


@dataclass(frozen=True)
class HybridModelConfig:
    backbone_name: str = "resnet34"
    pretrained_backbone: bool = False
    use_coord_channels: bool = True
    include_aspect: bool = False
    param_hidden_dim: int = 256
    residual_hidden_dim: int = 128
    residual_max_disp: float = 8.0
    return_pred_image: bool = False


class HybridLensCorrectionModel(nn.Module):
    """Hybrid lens-correction model for parametric + residual prediction.

    Forward input:
        image: Tensor[B, 3, H, W], expected in [0, 1]

    Forward output dict keys:
        params: Tensor[B, 8 or 9]
        residual_flow: Tensor[B, 2, Hr, Wr]  # alias for low-res flow
        residual_flow_lowres: Tensor[B, 2, Hr, Wr]
        residual_flow_fullres: Tensor[B, 2, H, W]
        debug_stats: dict[str, Tensor]
        pred_image: Tensor[B, 3, H, W]  # optional, only when warp backend enabled
    """

    def __init__(
        self,
        config: HybridModelConfig | None = None,
        *,
        param_bounds: ParametricBounds | None = None,
        warp_backend: ModelWarpBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = config or HybridModelConfig()
        self.warp_backend = warp_backend

        self.coord_appender = CoordChannelAppender() if self.config.use_coord_channels else None
        input_channels = 6 if self.config.use_coord_channels else 3

        self.backbone = create_backbone(
            self.config.backbone_name,
            in_channels=input_channels,
            pretrained=self.config.pretrained_backbone,
        )

        out_ch = self.backbone.spec.out_channels
        self.param_head = ParametricHead(
            in_channels=out_ch["layer4"],
            hidden_dim=self.config.param_hidden_dim,
            include_aspect=self.config.include_aspect,
            bounds=param_bounds,
        )
        self.residual_head = ResidualFlowHead(
            feature_channels=out_ch,
            hidden_dim=self.config.residual_hidden_dim,
            max_disp=self.config.residual_max_disp,
        )

    @staticmethod
    def _debug_stats(params: Tensor, residual_flow_lowres: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            stats = {
                "params_abs_mean": params.abs().mean(),
                "params_abs_max": params.abs().amax(),
                "residual_abs_mean_px": residual_flow_lowres.abs().mean(),
                "residual_abs_max_px": residual_flow_lowres.abs().amax(),
            }
        return stats

    def forward(self, image: Tensor, *, return_pred_image: bool | None = None) -> dict[str, Any]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,3,H,W]")
        if image.shape[1] != 3:
            raise ValueError("image must have 3 channels (RGB)")

        x = self.coord_appender(image) if self.coord_appender is not None else image
        features = self.backbone(x)

        params = self.param_head(features["layer4"])
        residual_flow_lowres = self.residual_head(features)
        residual_flow_fullres = F.interpolate(
            residual_flow_lowres,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        outputs: dict[str, Any] = {
            "params": params,
            "residual_flow": residual_flow_lowres,
            "residual_flow_lowres": residual_flow_lowres,
            "residual_flow_fullres": residual_flow_fullres,
            "debug_stats": self._debug_stats(params, residual_flow_lowres),
        }

        want_pred = self.config.return_pred_image if return_pred_image is None else bool(return_pred_image)
        if want_pred and self.warp_backend is not None:
            warp_out = self.warp_backend(image, params, residual_flow_lowres)
            if "pred_image" not in warp_out:
                raise ValueError("warp backend must return key 'pred_image' when return_pred_image=True")
            outputs.update(warp_out)

        return outputs
