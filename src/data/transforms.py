from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _to_hw(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("Expected (H, W) pair for transform size.")
    return int(value[0]), int(value[1])


@dataclass
class PairedTransformConfig:
    resize_hw: Optional[Tuple[int, int]] = None
    center_crop_hw: Optional[Tuple[int, int]] = None
    random_crop_hw: Optional[Tuple[int, int]] = None
    hflip_prob: float = 0.0
    seed: int = 123


class PairedTransforms:
    def __init__(self, cfg):
        self.resize_hw = _to_hw(getattr(cfg, "resize_hw", None))
        self.center_crop_hw = _to_hw(getattr(cfg, "center_crop_hw", None))
        self.random_crop_hw = _to_hw(getattr(cfg, "random_crop_hw", None))
        self.hflip_prob = float(getattr(cfg, "hflip_prob", 0.0))
        self.seed = int(getattr(cfg, "seed", 123))

        if not (0.0 <= self.hflip_prob <= 1.0):
            raise ValueError("hflip_prob must be within [0, 1].")

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)

    @staticmethod
    def _resize(x: torch.Tensor, y: torch.Tensor, size_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=True).squeeze(0)
        y = F.interpolate(y.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=True).squeeze(0)
        return x, y

    @staticmethod
    def _center_crop(
        x: torch.Tensor,
        y: torch.Tensor,
        crop_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        ch, cw = crop_hw
        if ch > h or cw > w:
            raise ValueError(f"Center crop {crop_hw} exceeds tensor size {(h, w)}.")

        top = (h - ch) // 2
        left = (w - cw) // 2
        return x[:, top : top + ch, left : left + cw], y[:, top : top + ch, left : left + cw]

    def _random_crop(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        crop_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        ch, cw = crop_hw
        if ch > h or cw > w:
            raise ValueError(f"Random crop {crop_hw} exceeds tensor size {(h, w)}.")

        max_top = h - ch
        max_left = w - cw
        top = int(torch.randint(0, max_top + 1, (1,), generator=self._generator).item())
        left = int(torch.randint(0, max_left + 1, (1,), generator=self._generator).item())
        return x[:, top : top + ch, left : left + cw], y[:, top : top + ch, left : left + cw]

    def _hflip(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        should_flip = torch.rand((1,), generator=self._generator).item() < self.hflip_prob
        if not should_flip:
            return x, y
        return x.flip(-1), y.flip(-1)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.resize_hw is not None:
            x, y = self._resize(x, y, self.resize_hw)
        if self.center_crop_hw is not None:
            x, y = self._center_crop(x, y, self.center_crop_hw)
        if self.random_crop_hw is not None:
            x, y = self._random_crop(x, y, self.random_crop_hw)
        if self.hflip_prob > 0.0:
            x, y = self._hflip(x, y)
        return x.contiguous(), y.contiguous()
