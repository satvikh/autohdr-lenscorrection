from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, OneCycleLR


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "none"  # one of: none, cosine, onecycle
    min_lr: float = 1e-6
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4


@dataclass
class SchedulerBundle:
    scheduler: LRScheduler | None
    step_interval: str  # "batch" or "epoch" or "none"


def create_optimizer(model: nn.Module, config: OptimConfig) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    total_steps: int,
    steps_per_epoch: int,
    max_lr: float | None = None,
) -> SchedulerBundle:
    name = config.name.lower()
    if name == "none":
        return SchedulerBundle(scheduler=None, step_interval="none")

    if name == "cosine":
        t_max = max(total_steps, 1)
        sched = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.min_lr)
        return SchedulerBundle(scheduler=sched, step_interval="batch")

    if name == "onecycle":
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive for onecycle scheduler")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive for onecycle scheduler")

        peak_lr = float(max_lr if max_lr is not None else optimizer.param_groups[0]["lr"])
        sched = OneCycleLR(
            optimizer,
            max_lr=peak_lr,
            total_steps=total_steps,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
        )
        return SchedulerBundle(scheduler=sched, step_interval="batch")

    raise ValueError(f"Unsupported scheduler name: {config.name}")