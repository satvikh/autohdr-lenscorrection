from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualFlowHead(nn.Module):
    """Predict bounded low-resolution residual flow from multi-scale features.

    Expected feature keys:
        - layer2: Tensor[B, C2, H/8,  W/8]
        - layer3: Tensor[B, C3, H/16, W/16]
        - layer4: Tensor[B, C4, H/32, W/32]

    Output:
        residual_flow_lowres: Tensor[B, 2, H/8, W/8] in pixel displacement units.
    """

    def __init__(
        self,
        feature_channels: dict[str, int],
        *,
        hidden_dim: int = 128,
        max_disp: float = 8.0,
    ) -> None:
        super().__init__()
        self.max_disp = float(max_disp)

        for key in ("layer2", "layer3", "layer4"):
            if key not in feature_channels:
                raise ValueError(f"feature_channels missing required key: {key}")

        self.lat2 = nn.Conv2d(feature_channels["layer2"], hidden_dim, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(feature_channels["layer3"], hidden_dim, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(feature_channels["layer4"], hidden_dim, kernel_size=1, bias=False)

        self.refine3 = _ConvBlock(hidden_dim, hidden_dim)
        self.refine2 = _ConvBlock(hidden_dim, hidden_dim)
        self.out_block = _ConvBlock(hidden_dim, hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, 2, kernel_size=3, stride=1, padding=1, bias=True)

        self._init_out_zero()

    def _init_out_zero(self) -> None:
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        for key in ("layer2", "layer3", "layer4"):
            if key not in features:
                raise ValueError(f"features missing required key: {key}")

        l2 = features["layer2"]
        l3 = features["layer3"]
        l4 = features["layer4"]

        if not (l2.ndim == l3.ndim == l4.ndim == 4):
            raise ValueError("layer2/layer3/layer4 features must all be 4D BCHW tensors")

        p4 = self.lat4(l4)

        p3 = self.lat3(l3) + F.interpolate(
            p4,
            size=l3.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        p3 = self.refine3(p3)

        p2 = self.lat2(l2) + F.interpolate(
            p3,
            size=l2.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        p2 = self.refine2(p2)

        h = self.out_block(p2)
        flow_raw = self.out_conv(h)
        flow = torch.tanh(flow_raw) * self.max_disp
        return flow