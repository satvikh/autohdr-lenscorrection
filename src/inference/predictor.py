from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.geometry.jacobian import jacobian_stats
from src.geometry.parametric_warp import build_parametric_grid
from src.geometry.residual_fusion import (
    adapt_residual_flow_to_bhwc,
    pixel_flow_to_normalized_delta,
    fuse_grids,
    upsample_residual_flow,
)
from src.geometry.warp_ops import warp_image
from src.inference.fallback import make_conservative_param_only
from src.inference.safety import SafetyConfig, evaluate_safety


ModelCallable = Callable[[Tensor], dict[str, Tensor]]


@dataclass
class PredictorConfig:
    """Full-resolution predictor config for hybrid-safe baseline inference."""

    resize_to: tuple[int, int] | None = None
    align_corners: bool = True
    padding_mode: str = "border"
    mode: str = "bilinear"
    safety_config: SafetyConfig | None = None


class Predictor:
    """Full-resolution predictor with optional hybrid residual path + fallback."""

    def __init__(self, model: ModelCallable, config: PredictorConfig | None = None):
        self.model = model
        self.config = config or PredictorConfig()

        if self.config.align_corners is not True:
            raise ValueError("align_corners must be True per global geometry policy")

    @staticmethod
    def _load_image_as_tensor(image_path: str | Path) -> Tensor:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    @staticmethod
    def _maybe_resize(image: Tensor, resize_to: tuple[int, int] | None) -> Tensor:
        if resize_to is None:
            return image
        target_h, target_w = resize_to
        return torch.nn.functional.interpolate(
            image,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )

    @staticmethod
    def _validate_params(params: Tensor, batch: int) -> None:
        if params.ndim != 2 or params.shape[0] != batch or params.shape[1] < 8:
            raise ValueError("params must have shape [B, 8+] and batch must match input image")

    def _param_grid(self, params: Tensor, image: Tensor) -> Tensor:
        _, _, h, w = image.shape
        return build_parametric_grid(
            params=params,
            height=h,
            width=w,
            align_corners=True,
            device=image.device,
            dtype=image.dtype,
        )

    def _hybrid_grid_and_residual(
        self,
        param_grid: Tensor,
        residual_flow: Tensor,
        out_h: int,
        out_w: int,
    ) -> tuple[Tensor, Tensor]:
        residual_bhwc_px = adapt_residual_flow_to_bhwc(residual_flow)
        residual_norm_lr = pixel_flow_to_normalized_delta(residual_bhwc_px)
        residual_norm_full = upsample_residual_flow(
            residual_norm_lr,
            target_h=out_h,
            target_w=out_w,
            align_corners=True,
        )
        hybrid_grid = fuse_grids(param_grid, residual_norm_full)
        return hybrid_grid, residual_norm_full

    def _evaluate(self, grid: Tensor, residual_norm: Tensor | None, cfg: SafetyConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        jac = jacobian_stats(grid)
        safety = evaluate_safety(grid, residual_flow_norm_bhwc=residual_norm, config=cfg)
        return jac, safety

    def predict(self, image_path: str | Path) -> tuple[Tensor, dict[str, Any]]:
        image = self._load_image_as_tensor(image_path).to(dtype=torch.float32)
        model_input = self._maybe_resize(image, self.config.resize_to)

        model_out = self.model(model_input)
        if not isinstance(model_out, dict) or "params" not in model_out:
            raise ValueError("model callable must return a dict containing key 'params'")

        params = model_out["params"]
        if not torch.is_tensor(params):
            raise ValueError("model output 'params' must be a torch.Tensor")
        params = params.to(device=image.device, dtype=image.dtype)

        batch, _, out_h, out_w = image.shape
        self._validate_params(params, batch)
        safety_cfg = self.config.safety_config or SafetyConfig()

        param_grid = self._param_grid(params, image)

        residual_raw = model_out.get("residual_flow")
        has_residual = torch.is_tensor(residual_raw)

        warnings: list[str] = []
        mode_used = "param_only"

        initial_safety: dict[str, Any]
        initial_jacobian: dict[str, Any]

        # Candidate 1: hybrid (only if residual exists)
        if has_residual:
            residual_t = residual_raw.to(device=image.device, dtype=image.dtype)
            hybrid_grid, residual_norm_full = self._hybrid_grid_and_residual(param_grid, residual_t, out_h, out_w)
            jac_h, safe_h = self._evaluate(hybrid_grid, residual_norm_full, safety_cfg)
            initial_safety = safe_h
            initial_jacobian = jac_h

            if safe_h["safe"]:
                mode_used = "hybrid"
                final_grid = hybrid_grid
                final_jacobian = jac_h
                final_safety = safe_h
            else:
                warnings.append("HYBRID_UNSAFE_FALLBACK_TO_PARAM_ONLY")
                jac_p, safe_p = self._evaluate(param_grid, None, safety_cfg)
                if safe_p["safe"]:
                    mode_used = "param_only"
                    final_grid = param_grid
                    final_jacobian = jac_p
                    final_safety = safe_p
                else:
                    warnings.append("PARAM_ONLY_UNSAFE_FALLBACK_TO_CONSERVATIVE")
                    conservative_params = make_conservative_param_only(params)
                    conservative_grid = self._param_grid(conservative_params, image)
                    jac_c, safe_c = self._evaluate(conservative_grid, None, safety_cfg)
                    mode_used = "param_only_conservative"
                    final_grid = conservative_grid
                    final_jacobian = jac_c
                    final_safety = safe_c
                    if not safe_c["safe"]:
                        warnings.append("HARD_UNSAFE_OUTPUT")
        else:
            # Candidate 1 (no residual available): param-only
            jac_p, safe_p = self._evaluate(param_grid, None, safety_cfg)
            initial_safety = safe_p
            initial_jacobian = jac_p

            if safe_p["safe"]:
                mode_used = "param_only"
                final_grid = param_grid
                final_jacobian = jac_p
                final_safety = safe_p
            else:
                warnings.append("PARAM_ONLY_UNSAFE_FALLBACK_TO_CONSERVATIVE")
                conservative_params = make_conservative_param_only(params)
                conservative_grid = self._param_grid(conservative_params, image)
                jac_c, safe_c = self._evaluate(conservative_grid, None, safety_cfg)
                mode_used = "param_only_conservative"
                final_grid = conservative_grid
                final_jacobian = jac_c
                final_safety = safe_c
                if not safe_c["safe"]:
                    warnings.append("HARD_UNSAFE_OUTPUT")

        warped = warp_image(
            image=image,
            grid=final_grid,
            mode=self.config.mode,
            padding_mode=self.config.padding_mode,
            align_corners=True,
        )

        metadata: dict[str, Any] = {
            "mode_used": mode_used,
            "safety": final_safety,
            "jacobian": final_jacobian,
            "warnings": warnings,
            "input_path": str(image_path),
            "input_shape": tuple(image.shape),
            "output_shape": tuple(warped.shape),
            "min": float(warped.min().item()),
            "max": float(warped.max().item()),
            "mean": float(warped.mean().item()),
            "initial_safety": initial_safety,
            "initial_jacobian": initial_jacobian,
        }

        return warped, metadata


def predict(
    image_path: str | Path,
    model: ModelCallable,
    config: PredictorConfig | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    predictor = Predictor(model=model, config=config)
    return predictor.predict(image_path)
