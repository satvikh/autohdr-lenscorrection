from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.geometry.parametric_warp import build_parametric_grid
from src.geometry.warp_ops import warp_image


ModelCallable = Callable[[Tensor], dict[str, Tensor]]


@dataclass
class PredictorConfig:
    """Baseline predictor config for param-only inference."""

    resize_to: tuple[int, int] | None = None
    align_corners: bool = True
    padding_mode: str = "border"
    mode: str = "bilinear"


class Predictor:
    """Minimal full-resolution param-only inference pipeline."""

    def __init__(self, model: ModelCallable, config: PredictorConfig | None = None):
        self.model = model
        self.config = config or PredictorConfig()

        if self.config.align_corners is not True:
            raise ValueError("align_corners must be True per global geometry policy")

    @staticmethod
    def _load_image_as_tensor(image_path: str | Path) -> Tensor:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0  # [H,W,C]

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,C,H,W]
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

    def predict(self, image_path: str | Path) -> tuple[Tensor, dict[str, Any]]:
        image = self._load_image_as_tensor(image_path).to(dtype=torch.float32)

        input_for_model = self._maybe_resize(image, self.config.resize_to)

        model_out = self.model(input_for_model)
        if not isinstance(model_out, dict) or "params" not in model_out:
            raise ValueError("model callable must return a dict containing key 'params'")

        params = model_out["params"]
        if not torch.is_tensor(params):
            raise ValueError("model output 'params' must be a torch.Tensor")

        batch, _, out_h, out_w = image.shape

        if params.ndim != 2 or params.shape[0] != batch:
            raise ValueError("params must have shape [B, 8+] and batch must match input image")

        grid = build_parametric_grid(
            params=params,
            height=out_h,
            width=out_w,
            align_corners=True,
            device=image.device,
            dtype=image.dtype,
        )

        warped = warp_image(
            image=image,
            grid=grid,
            mode=self.config.mode,
            padding_mode=self.config.padding_mode,
            align_corners=True,
        )

        metadata: dict[str, Any] = {
            "mode_used": "param_only",
            "safe": True,
            "input_path": str(image_path),
            "input_shape": tuple(image.shape),
            "output_shape": tuple(warped.shape),
            "min": float(warped.min().item()),
            "max": float(warped.max().item()),
            "mean": float(warped.mean().item()),
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
