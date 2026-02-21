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
from src.inference.fallback import make_conservative_param_only
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
    """Full-resolution param-only baseline with safety and conservative fallback."""

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
        batch = image.shape[0]
        if params.ndim != 2 or params.shape[0] != batch or params.shape[1] < 8:
            raise ValueError("params must have shape [B, 8+] and batch must match input image")

        safety_cfg = self.config.safety_config or SafetyConfig()

        candidate_grid = self._grid_from_params(params, image)
        candidate_jac = jacobian_stats(candidate_grid)
        candidate_safety = evaluate_safety(candidate_grid, residual_flow_norm_bhwc=None, config=safety_cfg)

        mode_used = "param_only"
        warnings: list[str] = []
        chosen_params = params
        final_grid = candidate_grid
        final_jac = candidate_jac
        final_safety = candidate_safety

        if not candidate_safety["safe"]:
            warnings.append("PARAM_ONLY_UNSAFE_FALLBACK_TO_CONSERVATIVE")
            mode_used = "param_only_conservative"
            chosen_params = make_conservative_param_only(params)

            final_grid = self._grid_from_params(chosen_params, image)
            final_jac = jacobian_stats(final_grid)
            final_safety = evaluate_safety(final_grid, residual_flow_norm_bhwc=None, config=safety_cfg)

            if not final_safety["safe"]:
                warnings.append("HARD_UNSAFE_OUTPUT")

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
            "initial_safety": candidate_safety,
            "initial_jacobian": candidate_jac,
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
