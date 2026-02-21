from __future__ import annotations

import torch
from torch import Tensor

from src.geometry.coords import make_identity_grid
from src.geometry.parametric_warp import build_parametric_grid
from src.geometry.residual_fusion import (
    adapt_residual_flow_to_bhwc,
    fuse_grids,
    pixel_flow_to_normalized_delta,
    upsample_residual_flow,
)
from src.geometry.warp_ops import warp_image


class MockWarpBackend:
    """Deterministic differentiable mock backend for training smoke tests.

    The mock uses:
    - small param-based global shifts from params[:, 5:7]
    - optional residual flow fusion if provided
    """

    def __init__(self, *, align_corners: bool = True) -> None:
        if align_corners is not True:
            raise ValueError("align_corners must be True")
        self.align_corners = align_corners

    def warp(
        self,
        image: Tensor,
        params: Tensor,
        residual_flow_lowres: Tensor | None,
    ) -> dict[str, Tensor]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        if params.ndim != 2:
            raise ValueError("params must have shape [B,N]")

        b, _, h, w = image.shape
        if params.shape[0] != b:
            raise ValueError("params batch size must match image batch size")

        param_grid = make_identity_grid(b, h, w, device=image.device, dtype=image.dtype)

        # Small learnable global shifts to keep params path differentiable.
        shift_x = 0.1 * torch.tanh(params[:, 5]).view(b, 1, 1)
        shift_y = 0.1 * torch.tanh(params[:, 6]).view(b, 1, 1)
        param_grid[..., 0] = param_grid[..., 0] + shift_x
        param_grid[..., 1] = param_grid[..., 1] + shift_y

        residual_norm_full: Tensor | None = None
        final_grid = param_grid

        if residual_flow_lowres is not None:
            residual_bhwc_px = adapt_residual_flow_to_bhwc(residual_flow_lowres)
            residual_norm_lr = pixel_flow_to_normalized_delta(residual_bhwc_px)
            residual_norm_full = upsample_residual_flow(
                residual_norm_lr,
                target_h=h,
                target_w=w,
                align_corners=True,
            )
            final_grid = fuse_grids(param_grid, residual_norm_full)

        pred_image = warp_image(
            image=image,
            grid=final_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        out: dict[str, Tensor] = {
            "pred_image": pred_image,
            "param_grid": param_grid,
            "final_grid": final_grid,
        }
        if residual_norm_full is not None:
            out["residual_flow_fullres_norm"] = residual_norm_full
        return out


class Person1GeometryWarpBackend:
    """Adapter backend using Person 1 geometry contracts directly."""

    def __init__(self, *, align_corners: bool = True) -> None:
        if align_corners is not True:
            raise ValueError("align_corners must be True")
        self.align_corners = align_corners

    def warp(
        self,
        image: Tensor,
        params: Tensor,
        residual_flow_lowres: Tensor | None,
    ) -> dict[str, Tensor]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        if params.ndim != 2:
            raise ValueError("params must have shape [B,N]")

        _, _, h, w = image.shape
        param_grid = build_parametric_grid(
            params=params,
            height=h,
            width=w,
            align_corners=True,
            device=image.device,
            dtype=image.dtype,
        )

        residual_norm_full: Tensor | None = None
        final_grid = param_grid

        if residual_flow_lowres is not None:
            residual_bhwc_px = adapt_residual_flow_to_bhwc(residual_flow_lowres)
            residual_norm_lr = pixel_flow_to_normalized_delta(residual_bhwc_px)
            residual_norm_full = upsample_residual_flow(
                residual_norm_lr,
                target_h=h,
                target_w=w,
                align_corners=True,
            )
            final_grid = fuse_grids(param_grid, residual_norm_full)

        pred_image = warp_image(
            image=image,
            grid=final_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        out: dict[str, Tensor] = {
            "pred_image": pred_image,
            "param_grid": param_grid,
            "final_grid": final_grid,
        }
        if residual_norm_full is not None:
            out["residual_flow_fullres_norm"] = residual_norm_full
        return out