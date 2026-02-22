from __future__ import annotations

from dataclasses import dataclass
from math import atanh
from typing import Dict

import torch
from torch import Tensor
from torch import nn


@dataclass(frozen=True)
class ParametricBounds:
    """Parameter bounds for bounded global lens outputs."""

    k1: tuple[float, float] = (-0.6, 0.6)
    k2: tuple[float, float] = (-0.3, 0.3)
    k3: tuple[float, float] = (-0.15, 0.15)
    p1: tuple[float, float] = (-0.03, 0.03)
    p2: tuple[float, float] = (-0.03, 0.03)
    dcx: tuple[float, float] = (-0.08, 0.08)
    dcy: tuple[float, float] = (-0.08, 0.08)
    scale: tuple[float, float] = (0.90, 1.20)
    aspect: tuple[float, float] = (0.97, 1.03)


class ParametricHead(nn.Module):
    """Predict bounded global lens parameters from high-level features.

    Input:
        feat: Tensor[B, C, Hf, Wf]

    Output:
        params: Tensor[B, N]
            Order (N=8): [k1, k2, k3, p1, p2, dcx, dcy, scale]
            Order (N=9 when include_aspect=True): [..., aspect]
    """

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int = 256,
        include_aspect: bool = False,
        dropout_p: float = 0.0,
        bounds: ParametricBounds | None = None,
    ) -> None:
        super().__init__()
        self.include_aspect = bool(include_aspect)
        self.bounds = bounds or ParametricBounds()

        self.param_names = ["k1", "k2", "k3", "p1", "p2", "dcx", "dcy", "scale"]
        if self.include_aspect:
            self.param_names.append("aspect")

        out_dim = len(self.param_names)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_near_identity()

    def _bounds_map(self) -> Dict[str, tuple[float, float]]:
        m = {
            "k1": self.bounds.k1,
            "k2": self.bounds.k2,
            "k3": self.bounds.k3,
            "p1": self.bounds.p1,
            "p2": self.bounds.p2,
            "dcx": self.bounds.dcx,
            "dcy": self.bounds.dcy,
            "scale": self.bounds.scale,
            "aspect": self.bounds.aspect,
        }
        return {k: m[k] for k in self.param_names}

    def _identity_target(self, name: str) -> float:
        if name in {"scale", "aspect"}:
            return 1.0
        return 0.0

    def _init_near_identity(self) -> None:
        final_linear = self.mlp[-1]
        assert isinstance(final_linear, nn.Linear)

        nn.init.zeros_(final_linear.weight)
        bias = torch.zeros_like(final_linear.bias)

        bounds_map = self._bounds_map()
        for i, name in enumerate(self.param_names):
            lo, hi = bounds_map[name]
            if not (hi > lo):
                raise ValueError(f"Invalid bounds for {name}: {(lo, hi)}")

            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            target = self._identity_target(name)

            alpha = (target - center) / max(half, 1e-12)
            alpha = float(max(min(alpha, 0.999999), -0.999999))
            bias[i] = float(atanh(alpha))

        with torch.no_grad():
            final_linear.bias.copy_(bias)

    def apply_bounds(self, raw: Tensor) -> Tensor:
        if raw.ndim != 2:
            raise ValueError("raw must have shape [B,N]")

        bounds_map = self._bounds_map()
        outs: list[Tensor] = []

        for i, name in enumerate(self.param_names):
            lo, hi = bounds_map[name]
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            bounded = center + half * torch.tanh(raw[:, i])
            outs.append(bounded)

        return torch.stack(outs, dim=1)

    def predict_raw(self, feat: Tensor) -> Tensor:
        if feat.ndim != 4:
            raise ValueError("feat must have shape [B,C,H,W]")

        pooled = self.pool(feat)
        return self.mlp(pooled)

    def forward(self, feat: Tensor) -> Tensor:
        raw = self.predict_raw(feat)
        return self.apply_bounds(raw)
