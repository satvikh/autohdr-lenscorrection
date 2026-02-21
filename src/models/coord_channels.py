from __future__ import annotations

import torch
from torch import Tensor
from torch import nn


class CoordChannelAppender(nn.Module):
    """Append normalized coordinate channels to image tensors.

    Input:
        image: Tensor[B, C, H, W], typically RGB in range [0, 1].

    Output:
        Tensor[B, C + 3, H, W] where appended channels are:
            - x: normalized horizontal coordinates in [-1, 1]
            - y: normalized vertical coordinates in [-1, 1]
            - r: radial distance sqrt(x^2 + y^2), in [0, sqrt(2)]
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    @staticmethod
    def _coord_grid(
        batch: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")

        if width > 1:
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            xs = torch.zeros(1, device=device, dtype=dtype)

        if height > 1:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        else:
            ys = torch.zeros(1, device=device, dtype=dtype)

        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        r_grid = torch.sqrt((x_grid * x_grid) + (y_grid * y_grid) + 1e-12)

        coords = torch.stack((x_grid, y_grid, r_grid), dim=0).unsqueeze(0)
        return coords.expand(batch, -1, -1, -1).contiguous()

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")

        batch, _, height, width = image.shape
        coords = self._coord_grid(
            batch,
            height,
            width,
            device=image.device,
            dtype=image.dtype,
        )
        return torch.cat((image, coords), dim=1)