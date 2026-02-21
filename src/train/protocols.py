from __future__ import annotations

from typing import Protocol

from torch import Tensor


class WarpBackend(Protocol):
    """Protocol for dependency-injected image warping backends.

    Args:
        image: Tensor[B, C, H, W]
        params: Tensor[B, N]
        residual_flow_lowres: optional Tensor[B, 2, Hr, Wr]

    Returns:
        Dict with at least:
            pred_image: Tensor[B, C, H, W]
        Optional keys:
            param_grid: Tensor[B, H, W, 2]
            final_grid: Tensor[B, H, W, 2]
            residual_flow_fullres_norm: Tensor[B, H, W, 2]
    """

    def warp(
        self,
        image: Tensor,
        params: Tensor,
        residual_flow_lowres: Tensor | None,
    ) -> dict[str, Tensor]:
        ...