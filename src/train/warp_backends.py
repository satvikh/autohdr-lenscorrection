from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.geometry.coords import make_identity_grid
from src.geometry.jacobian import jacobian_stats
from src.geometry.parametric_warp import build_parametric_grid
from src.geometry.residual_fusion import (
    adapt_residual_flow_to_bhwc,
    fuse_grids,
    pixel_flow_to_normalized_delta,
    upsample_residual_flow,
)
from src.geometry.warp_ops import warp_image
from src.inference.safety import SafetyConfig, evaluate_safety


@dataclass(frozen=True)
class WarpBackendConfig:
    """Config for train-time warp backend diagnostics.

    Attributes:
        align_corners: must remain True to match global geometry policy.
        mode: interpolation mode for `grid_sample`.
        padding_mode: padding mode for `grid_sample`.
        include_safety_metrics: when True, run Person-1 safety evaluation and include report.
    """

    align_corners: bool = True
    mode: str = "bilinear"
    padding_mode: str = "border"
    include_safety_metrics: bool = True


class _BaseBackend:
    def __init__(self, config: WarpBackendConfig | None = None) -> None:
        self.config = config or WarpBackendConfig()
        if self.config.align_corners is not True:
            raise ValueError("align_corners must be True")

    @staticmethod
    def _oob_ratio(grid_bhwc: Tensor) -> Tensor:
        x = grid_bhwc[..., 0]
        y = grid_bhwc[..., 1]
        oob = (x < -1.0) | (x > 1.0) | (y < -1.0) | (y > 1.0)
        return oob.float().mean()

    def _make_stats(self, final_grid: Tensor, residual_norm_full: Tensor | None) -> dict[str, Any]:
        jac = jacobian_stats(final_grid)

        stats: dict[str, Any] = {
            "oob_ratio": float(self._oob_ratio(final_grid).item()),
            "negative_det_pct": float(jac["negative_det_pct"]),
            "det_min": float(jac["det_min"]),
            "det_p01": float(jac["det_p01"]),
            "det_mean": float(jac["det_mean"]),
            "high_grad_area_frac": float(jac.get("high_grad_area_frac", 0.0)),
        }

        if residual_norm_full is not None:
            res_abs = residual_norm_full.abs()
            stats["residual_abs_mean_norm"] = float(res_abs.mean().item())
            stats["residual_abs_max_norm"] = float(res_abs.max().item())
        else:
            stats["residual_abs_mean_norm"] = 0.0
            stats["residual_abs_max_norm"] = 0.0

        if self.config.include_safety_metrics:
            safety_cfg = SafetyConfig()
            safety = evaluate_safety(final_grid, residual_flow_norm_bhwc=residual_norm_full, config=safety_cfg)
            stats["safety"] = safety

        return stats

    def _warp_image(self, image: Tensor, final_grid: Tensor) -> Tensor:
        return warp_image(
            image=image,
            grid=final_grid,
            mode=self.config.mode,
            padding_mode=self.config.padding_mode,
            align_corners=True,
        )


class MockWarpBackend(_BaseBackend):
    """Deterministic differentiable mock backend for train-step smoke tests.

    The mock keeps gradients through both parameter and residual branches.
    """

    def warp(
        self,
        image: Tensor,
        params: Tensor,
        residual_flow_lowres: Tensor | None,
    ) -> dict[str, Any]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        if params.ndim != 2:
            raise ValueError("params must have shape [B,N]")

        b, _, h, w = image.shape
        if params.shape[0] != b:
            raise ValueError("params batch size must match image batch size")

        param_grid = make_identity_grid(b, h, w, device=image.device, dtype=image.dtype)

        # Small param-conditioned shifts keep this path differentiable in smoke tests.
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

        pred_image = self._warp_image(image, final_grid)
        stats = self._make_stats(final_grid, residual_norm_full)

        out: dict[str, Any] = {
            "pred_image": pred_image,
            "param_grid": param_grid,
            "final_grid": final_grid,
            "warp_stats": stats,
        }
        if residual_norm_full is not None:
            out["residual_flow_fullres_norm"] = residual_norm_full
        return out


class Person1GeometryWarpBackend(_BaseBackend):
    """Adapter backend using Person 1 geometry contracts directly."""

    def warp(
        self,
        image: Tensor,
        params: Tensor,
        residual_flow_lowres: Tensor | None,
    ) -> dict[str, Any]:
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

        pred_image = self._warp_image(image, final_grid)
        stats = self._make_stats(final_grid, residual_norm_full)

        out: dict[str, Any] = {
            "pred_image": pred_image,
            "param_grid": param_grid,
            "final_grid": final_grid,
            "warp_stats": stats,
        }
        if residual_norm_full is not None:
            out["residual_flow_fullres_norm"] = residual_norm_full
        return out