from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def warp_image(
    image: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> Tensor:
    """Warp image using a backward BHWC sampling grid.

    - image: [B, C, H, W]
    - grid: [B, H, W, 2], normalized to [-1, 1], (x, y) order
    """
    if align_corners is not True:
        raise ValueError("align_corners must be True per global geometry policy")

    if image.ndim != 4:
        raise ValueError("image must have shape [B, C, H, W]")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [B, H, W, 2]")

    b_i, _, h_i, w_i = image.shape
    b_g, h_g, w_g, _ = grid.shape

    if b_i != b_g:
        raise ValueError("image and grid batch size must match")
    if h_i != h_g or w_i != w_g:
        raise ValueError("grid spatial size must match image spatial size")

    if not torch.is_floating_point(grid):
        grid = grid.float()

    return F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
