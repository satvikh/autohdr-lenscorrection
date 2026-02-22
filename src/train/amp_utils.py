from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def autocast_context(*, enabled: bool, device: torch.device) -> Any:
    """Return autocast context for supported device types."""
    if not enabled:
        return nullcontext()

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if device.type == "cpu":
        # CPU autocast uses bfloat16 in modern PyTorch builds.
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def build_grad_scaler(*, enabled: bool, device: torch.device) -> torch.amp.GradScaler | None:
    """Create GradScaler when AMP is enabled on CUDA."""
    if not enabled or device.type != "cuda":
        return None
    return torch.amp.GradScaler(
        "cuda",
        init_scale=16384.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
