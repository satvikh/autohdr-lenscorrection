from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.losses.composite import CompositeLoss
from src.train.amp_utils import build_grad_scaler
from src.train.checkpointing import save_checkpoint
from src.train.logging_utils import RunningAverage, tensor_dict_to_float
from src.train.protocols import WarpBackend
from src.train.stage_configs import StageToggles
from src.train.train_step import run_eval_step, run_train_step


@dataclass(frozen=True)
class EngineConfig:
    epochs: int = 1
    amp_enabled: bool = False
    grad_clip_norm: float | None = 1.0
    log_interval: int = 10
    device: str = "cpu"
    max_steps_per_epoch: int | None = None
    max_val_steps: int | None = None
    checkpoint_dir: str = "outputs/runs"


class TrainerEngine:
    """Modular trainer engine for stage-based training."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: CompositeLoss,
        stage: StageToggles,
        warp_backend: WarpBackend,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        scheduler_step_interval: str,
        config: EngineConfig,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.stage = stage
        self.warp_backend = warp_backend
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_interval = scheduler_step_interval
        self.config = config

        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.scaler = build_grad_scaler(enabled=config.amp_enabled, device=self.device)

        self.global_step = 0
        self.best_val_loss: float | None = None

    def _maybe_step_scheduler_batch(self) -> None:
        if self.scheduler is not None and self.scheduler_step_interval == "batch":
            self.scheduler.step()

    def _maybe_step_scheduler_epoch(self) -> None:
        if self.scheduler is not None and self.scheduler_step_interval == "epoch":
            self.scheduler.step()

    def train_one_epoch(self, train_loader: Iterable[dict[str, Tensor]], epoch: int) -> dict[str, float]:
        tracker = RunningAverage()

        for step_idx, batch in enumerate(train_loader, start=1):
            if self.config.max_steps_per_epoch is not None and step_idx > self.config.max_steps_per_epoch:
                break

            out = run_train_step(
                model=self.model,
                batch=batch,
                loss_fn=self.loss_fn,
                warp_backend=self.warp_backend,
                stage=self.stage,
                optimizer=self.optimizer,
                scaler=self.scaler,
                amp_enabled=self.config.amp_enabled,
                grad_clip_norm=self.config.grad_clip_norm,
                device=self.device,
            )
            self.global_step += 1

            comp = tensor_dict_to_float(out.components)
            tracker.update(comp)

            self._maybe_step_scheduler_batch()

            if self.config.log_interval > 0 and step_idx % self.config.log_interval == 0:
                avg = tracker.averages()
                print(f"[train] epoch={epoch} step={step_idx} total={avg.get('total', float('nan')):.6f}")

        self._maybe_step_scheduler_epoch()
        return tracker.averages()

    def validate(self, val_loader: Iterable[dict[str, Tensor]], epoch: int) -> dict[str, float]:
        tracker = RunningAverage()

        for step_idx, batch in enumerate(val_loader, start=1):
            if self.config.max_val_steps is not None and step_idx > self.config.max_val_steps:
                break

            out = run_eval_step(
                model=self.model,
                batch=batch,
                loss_fn=self.loss_fn,
                warp_backend=self.warp_backend,
                stage=self.stage,
                amp_enabled=self.config.amp_enabled,
                device=self.device,
            )
            comp = tensor_dict_to_float(out.components)
            tracker.update(comp)

        avg = tracker.averages()
        print(f"[val] epoch={epoch} total={avg.get('total', float('nan')):.6f}")
        return avg

    def fit(
        self,
        *,
        train_loader: Iterable[dict[str, Tensor]],
        val_loader: Iterable[dict[str, Tensor]] | None = None,
        run_name: str = "default_run",
    ) -> dict[str, float]:
        final_train: dict[str, float] = {}
        final_val: dict[str, float] = {}

        ckpt_dir = Path(self.config.checkpoint_dir) / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            final_train = self.train_one_epoch(train_loader, epoch)

            if val_loader is not None:
                final_val = self.validate(val_loader, epoch)
                val_total = final_val.get("total", float("inf"))

                if self.best_val_loss is None or val_total < self.best_val_loss:
                    self.best_val_loss = val_total
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                        best_metric=val_total,
                        extra={"stage": self.stage.name},
                    )

            save_checkpoint(
                ckpt_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metric=self.best_val_loss,
                extra={"stage": self.stage.name},
            )

        return {
            "train_total": final_train.get("total", float("nan")),
            "val_total": final_val.get("total", float("nan")) if final_val else float("nan"),
            "best_val_total": float(self.best_val_loss) if self.best_val_loss is not None else float("nan"),
        }