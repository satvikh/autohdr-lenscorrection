from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor
from torch import nn


try:
    from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50

    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - environment-dependent
    _HAS_TORCHVISION = False


FeatureDict = Dict[str, Tensor]


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    out_channels: dict[str, int]


class TinyBackbone(nn.Module):
    """Small CNN backbone for smoke tests and environments without torchvision.

    Input:
        x: Tensor[B, C, H, W]

    Output feature dict keys and shapes:
        stem:   Tensor[B,  32, H/2,  W/2]
        layer1: Tensor[B,  64, H/4,  W/4]
        layer2: Tensor[B, 128, H/8,  W/8]
        layer3: Tensor[B, 192, H/16, W/16]
        layer4: Tensor[B, 256, H/32, W/32]
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._block(32, 64)
        self.layer2 = self._block(64, 128)
        self.layer3 = self._block(128, 192)
        self.layer4 = self._block(192, 256)

        self.spec = BackboneSpec(
            name="tiny",
            out_channels={
                "stem": 32,
                "layer1": 64,
                "layer2": 128,
                "layer3": 192,
                "layer4": 256,
            },
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> FeatureDict:
        stem = self.stem(x)
        l1 = self.layer1(stem)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return {"stem": stem, "layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}


class ResNetBackbone(nn.Module):
    """ResNet backbone wrapper returning multi-scale features.

    Supported names:
        - resnet34
        - resnet50

    Input:
        x: Tensor[B, C, H, W]

    Output:
        dict with keys stem/layer1/layer2/layer3/layer4.
    """

    def __init__(self, name: str, in_channels: int, *, pretrained: bool = False) -> None:
        super().__init__()

        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision is required for ResNet backbones. Use backbone='tiny' for smoke tests.")

        lname = name.lower()
        if lname == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
            ch = {"stem": 64, "layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        elif lname == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            ch = {"stem": 64, "layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.zero_()
                copy_ch = min(3, in_channels)
                new_conv.weight[:, :copy_ch] = model.conv1.weight[:, :copy_ch]
                if in_channels > 3:
                    mean_kernel = model.conv1.weight[:, :1].mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:in_channels] = mean_kernel
            model.conv1 = new_conv

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.spec = BackboneSpec(name=lname, out_channels=ch)

    def forward(self, x: Tensor) -> FeatureDict:
        x = self.conv1(x)
        x = self.bn1(x)
        stem = self.relu(x)

        x = self.maxpool(stem)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return {"stem": stem, "layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}


def create_backbone(
    name: str,
    *,
    in_channels: int,
    pretrained: bool = False,
) -> nn.Module:
    lname = name.lower()
    if lname == "tiny":
        return TinyBackbone(in_channels=in_channels)
    if lname in {"resnet34", "resnet50"}:
        return ResNetBackbone(name=lname, in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unknown backbone name: {name}")