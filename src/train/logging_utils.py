from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import Tensor


def tensor_dict_to_float(metrics: dict[str, Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if not torch.is_tensor(v):
            continue
        out[k] = float(v.detach().item())
    return out


@dataclass
class RunningAverage:
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def update(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def averages(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, s in self.sums.items():
            c = self.counts.get(key, 1)
            out[key] = s / max(c, 1)
        return out