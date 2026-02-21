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
from src.geometry.warp_ops import warp_image
from src.inference.fallback import run_fallback_hierarchy
from src.inference.safety import SafetyConfig, evaluate_safety


ModelCallable = Callable[[Tensor], dict[str, Tensor]]


@dataclass
class PredictorConfig:
    """Baseline predictor config for full-resolution param-only inference."""

    resize_to: tuple[int, int] | None = None
    align_corners: bool = True
    padding_mode: str = "border"
    mode: str = "bilinear"
    safety_config: SafetyConfig | None = None


class Predictor:
    """Full-resolution param-only baseline with safety and fallback utilities."""

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

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor

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

    def _grid_from_params(self, params: Tensor, image: Tensor) -> Tensor:
        _, _, out_h, out_w = image.shape
        return build_parametric_grid(
            params=params,
            height=out_h,
            width=out_w,
            align_corners=True,
            device=image.device,
            dtype=image.dtype,
        )

    def predict(self, image_path: str | Path) -> tuple[Tensor, dict[str, Any]]:
        image = self._load_image_as_tensor(image_path).to(dtype=torch.float32)
        input_for_model = self._maybe_resize(image, self.config.resize_to)

        model_out = self.model(input_for_model)
        if not isinstance(model_out, dict) or "params" not in model_out:
            raise ValueError("model callable must return a dict containing key 'params'")

        params = model_out["params"]
        if not torch.is_tensor(params):
            raise ValueError("model output 'params' must be a torch.Tensor")

        params = params.to(device=image.device, dtype=image.dtype)
        batch, _, _, _ = image.shape
        if params.ndim != 2 or params.shape[0] != batch or params.shape[1] < 8:
            raise ValueError("params must have shape [B, 8+] and batch must match input image")

        safety_cfg = self.config.safety_config or SafetyConfig()

        base_grid = self._grid_from_params(params, image)
        base_jac = jacobian_stats(base_grid)
        base_safety = evaluate_safety(base_grid, residual_flow_norm_bhwc=None, config=safety_cfg)

        mode_used = "param_only"
        chosen_params = params
        warnings: list[str] = []
        final_safety = base_safety

        if not base_safety["safe"]:
            def _safety_evaluator(candidate_params: Tensor, mode: str) -> dict[str, Any]:
                candidate_grid = self._grid_from_params(candidate_params, image)
                return evaluate_safety(candidate_grid, residual_flow_norm_bhwc=None, config=safety_cfg)

            mode_used, chosen_params, warnings, final_safety = run_fallback_hierarchy(
                hybrid_params=params,
                param_only_params=params,
                safety_evaluator=_safety_evaluator,
            )

            # Baseline predictor does not apply residual path yet.
            if mode_used == "hybrid":
                mode_used = "param_only"

        final_grid = self._grid_from_params(chosen_params, image)
        final_jac = jacobian_stats(final_grid)

        # Single-pass full-resolution warp.
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
            "jacobian": final_jac,
            "warnings": warnings,
            "input_path": str(image_path),
            "input_shape": tuple(image.shape),
            "output_shape": tuple(warped.shape),
            "min": float(warped.min().item()),
            "max": float(warped.max().item()),
            "mean": float(warped.mean().item()),
            "initial_safety": base_safety,
            "initial_jacobian": base_jac,
        }

        return warped, metadata


def predict(
    image_path: str | Path,
    model: ModelCallable,
    config: PredictorConfig | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Functional wrapper around Predictor for simple usage."""
    predictor = Predictor(model=model, config=config)
    return predictor.predict(image_path)
