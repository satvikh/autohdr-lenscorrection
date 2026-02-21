from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor


class WarpBackend(Protocol):
    """Protocol for dependency-injected image warping backends.

    Args:
        image: Tensor[B, C, H, W], input image in [0, 1].
        params: Tensor[B, N], bounded global lens parameters.
        residual_flow_lowres: optional residual flow Tensor[B,2,Hr,Wr] or Tensor[B,Hr,Wr,2].

    Returns:
        Dict with at least:
            pred_image: Tensor[B, C, H, W]
            warp_stats: dict[str, Any]
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
    ) -> dict[str, Any]:
        ...


class ProxyScorer(Protocol):
    """Protocol for optional proxy scorer integration (Person 3 owned).

    Expected callable signature:
        compute_proxy_score(pred, gt, config) -> dict
    """

    def __call__(self, pred: Tensor, gt: Tensor, config: Any | None = None) -> dict[str, Any]:
        ...
