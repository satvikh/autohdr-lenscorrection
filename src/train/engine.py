from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.losses.composite import CompositeLoss
from src.train.amp_utils import build_grad_scaler
from src.train.checkpointing import save_checkpoint
from src.train.debug_dump import dump_debug_triplet
from src.train.logging_utils import RunningAverage, tensor_dict_to_float
from src.train.protocols import ProxyScorer, WarpBackend
from src.train.proxy_hooks import compute_proxy_metrics_for_batch, resolve_proxy_scorer
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

    proxy_enabled: bool = False
    proxy_module_path: str | None = None
    proxy_function_name: str = "compute_proxy_score"
    proxy_config: Any | None = None

    debug_dump_dir: str | None = None
    debug_dump_max_images: int = 0
    fail_on_nonfinite_loss: bool = True
    param_saturation_warn_threshold: float = 0.50
    residual_warn_abs_max_px: float = 20.0
    best_metric_name: str = "total"
    best_metric_mode: str = "min"  # "min" or "max"


class TrainerEngine:
    """Modular trainer engine for stage-based training and validation."""

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

        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.scaler = build_grad_scaler(enabled=config.amp_enabled, device=self.device)

        self.global_step = 0
        self.best_val_loss: float | None = None

        self.proxy_scorer: ProxyScorer | None = None
        if self.config.proxy_enabled:
            self.proxy_scorer = resolve_proxy_scorer(
                module_path=self.config.proxy_module_path,
                function_name=self.config.proxy_function_name,
            )
            if self.proxy_scorer is None:
                print(
                    f"[warn] Proxy scorer enabled but unresolved: "
                    f"{self.config.proxy_module_path}.{self.config.proxy_function_name}"
                )

        self._debug_dump_written = 0
        self._warned_once: set[str] = set()
        mode = str(self.config.best_metric_mode).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError(f"best_metric_mode must be 'min' or 'max', got: {self.config.best_metric_mode}")
        self._best_metric_mode = mode

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        """Resolve an execution device with explicit fallbacks.

        If a CUDA/MPS device is requested but unavailable, training falls back to CPU
        and emits a warning message. This avoids hard failures in smoke/debug runs.
        """
        req = requested.lower().strip()
        if req.startswith("cuda") and not torch.cuda.is_available():
            print(f"[warn] Requested device '{requested}' is unavailable; falling back to 'cpu'.")
            return torch.device("cpu")
        if req.startswith("mps") and not torch.backends.mps.is_available():
            print(f"[warn] Requested device '{requested}' is unavailable; falling back to 'cpu'.")
            return torch.device("cpu")
        return torch.device(requested)

    def _maybe_step_scheduler_batch(self) -> None:
        if self.scheduler is not None and self.scheduler_step_interval == "batch":
            self.scheduler.step()

    def _maybe_step_scheduler_epoch(self) -> None:
        if self.scheduler is not None and self.scheduler_step_interval == "epoch":
            self.scheduler.step()

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _collect_base_metrics(self, out_components: dict[str, Tensor], diagnostics: dict[str, float]) -> dict[str, float]:
        metrics = tensor_dict_to_float(out_components)
        metrics.update(diagnostics)
        metrics["lr"] = self._current_lr()
        return metrics

    def _check_finite_train_total(self, total: float, *, epoch: int, step_idx: int) -> None:
        if torch.isfinite(torch.tensor(total)):
            return
        msg = (
            f"Non-finite training loss at epoch={epoch}, step={step_idx}. "
            "Check data ranges, LR, and loss weights."
        )
        if self.config.fail_on_nonfinite_loss:
            raise FloatingPointError(msg)
        print(f"[warn] {msg}")

    def _maybe_warn_saturation(self, metrics: dict[str, float], *, epoch: int, step_idx: int) -> None:
        sat_max = float(metrics.get("param_sat_frac_max", 0.0))
        if sat_max < float(self.config.param_saturation_warn_threshold):
            return
        key = f"sat:{epoch}"
        if key in self._warned_once:
            return
        self._warned_once.add(key)

        sat_items: list[tuple[str, float]] = []
        for k, v in metrics.items():
            if not k.startswith("param_sat_frac_") or k == "param_sat_frac_max":
                continue
            if float(v) >= float(self.config.param_saturation_warn_threshold):
                name = k.replace("param_sat_frac_", "")
                sat_items.append((name, float(v)))
        sat_items.sort(key=lambda x: x[1], reverse=True)
        sat_str = ", ".join(f"{name}:{frac:.2f}" for name, frac in sat_items[:4]) if sat_items else "none"
        print(
            f"[warn] Parameter saturation high at epoch={epoch}, step={step_idx}: "
            f"param_sat_frac_max={sat_max:.3f} (threshold={self.config.param_saturation_warn_threshold:.3f}); "
            f"hot_params=[{sat_str}]"
        )

    def _maybe_warn_residual(self, metrics: dict[str, float], *, epoch: int, step_idx: int) -> None:
        lowres_max = float(metrics.get("residual_lowres_abs_max_px", 0.0))
        fullres_norm_max = float(metrics.get("residual_fullres_norm_abs_max", 0.0))
        residual_max = float(
            max(
                lowres_max,
                fullres_norm_max,
            )
        )
        if residual_max < float(self.config.residual_warn_abs_max_px):
            return
        key = f"residual:{epoch}"
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        print(
            f"[warn] Residual magnitude unusually large at epoch={epoch}, step={step_idx}: "
            f"lowres_abs_max_px={lowres_max:.3f}, fullres_norm_abs_max={fullres_norm_max:.3f}, "
            f"threshold={self.config.residual_warn_abs_max_px:.3f}"
        )

    @staticmethod
    def _metric_summary(metrics: dict[str, float], keys: list[str]) -> str:
        parts: list[str] = []
        for k in keys:
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.4f}")
        return " ".join(parts)

    def _maybe_proxy_metrics(self, pred_image: Tensor, target_image: Tensor) -> dict[str, float]:
        if self.proxy_scorer is None:
            return {}

        try:
            return compute_proxy_metrics_for_batch(
                scorer=self.proxy_scorer,
                pred_batch=pred_image,
                target_batch=target_image,
                config=self.config.proxy_config,
            )
        except Exception:
            # Training must remain resilient even when proxy integration is unstable.
            return {"proxy_error": 1.0}

    def _maybe_debug_dump(self, *, prefix: str, input_image: Tensor, pred_image: Tensor, target_image: Tensor) -> None:
        if not self.config.debug_dump_dir:
            return
        if self._debug_dump_written >= max(int(self.config.debug_dump_max_images), 0):
            return

        out_dir = Path(self.config.debug_dump_dir)
        dump_debug_triplet(
            out_dir=out_dir,
            prefix=f"{prefix}_{self._debug_dump_written:04d}",
            input_image=input_image,
            pred_image=pred_image,
            target_image=target_image,
        )
        self._debug_dump_written += 1

    def _resolve_best_metric_from_val(self, val_metrics: dict[str, float]) -> float:
        key = str(self.config.best_metric_name).strip()
        candidate = float(val_metrics.get(key, float("nan")))
        if torch.isfinite(torch.tensor(candidate)):
            return candidate
        # Backward-compatible fallback keeps historical behavior.
        fallback = float(val_metrics.get("total", float("nan")))
        return fallback

    def _is_better_metric(self, candidate: float) -> bool:
        if not torch.isfinite(torch.tensor(candidate)):
            return False
        if self.best_val_loss is None:
            return True
        if self._best_metric_mode == "min":
            return candidate < self.best_val_loss
        return candidate > self.best_val_loss

    def train_one_epoch(self, train_loader: Iterable[dict[str, Any]], epoch: int) -> dict[str, float]:
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

            metrics = self._collect_base_metrics(out.components, out.diagnostics)
            self._check_finite_train_total(float(metrics.get("total", float("nan"))), epoch=epoch, step_idx=step_idx)
            self._maybe_warn_saturation(metrics, epoch=epoch, step_idx=step_idx)
            self._maybe_warn_residual(metrics, epoch=epoch, step_idx=step_idx)
            tracker.update(metrics)

            self._maybe_step_scheduler_batch()

            if self.config.log_interval > 0 and step_idx % self.config.log_interval == 0:
                avg = tracker.averages()
                loss_msg = self._metric_summary(
                    avg,
                    ["total", "pixel", "ssim", "edge", "grad_orient", "flow_tv", "flow_mag", "jacobian"],
                )
                diag_msg = self._metric_summary(
                    avg,
                    [
                        "param_sat_frac_max",
                        "residual_lowres_abs_mean_px",
                        "residual_lowres_abs_max_px",
                        "warp_oob_ratio",
                        "warp_negative_det_pct",
                        "warp_safety_safe",
                    ],
                )
                print(
                    f"[train] epoch={epoch} step={step_idx} "
                    f"{loss_msg} "
                    f"lr={avg.get('lr', float('nan')):.6e} "
                    f"{diag_msg}"
                )

        self._maybe_step_scheduler_epoch()
        return tracker.averages()

    def validate(self, val_loader: Iterable[dict[str, Any]], epoch: int) -> dict[str, float]:
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

            metrics = self._collect_base_metrics(out.components, out.diagnostics)

            if isinstance(batch.get("target_image"), torch.Tensor):
                target = batch["target_image"].to(device=out.pred_image.device, dtype=out.pred_image.dtype)
                proxy_metrics = self._maybe_proxy_metrics(out.pred_image, target)
                metrics.update(proxy_metrics)

                if isinstance(batch.get("input_image"), torch.Tensor):
                    self._maybe_debug_dump(
                        prefix=f"epoch{epoch}_step{step_idx}",
                        input_image=batch["input_image"].to(device=out.pred_image.device, dtype=out.pred_image.dtype),
                        pred_image=out.pred_image,
                        target_image=target,
                    )

            tracker.update(metrics)

        avg = tracker.averages()
        loss_msg = self._metric_summary(
            avg,
            ["total", "pixel", "ssim", "edge", "grad_orient", "flow_tv", "flow_mag", "jacobian"],
        )
        diag_msg = self._metric_summary(
            avg,
            ["warp_oob_ratio", "warp_negative_det_pct", "warp_safety_safe", "proxy_total_score", "proxy_hard_fail"],
        )
        print(
            f"[val] epoch={epoch} {loss_msg} {diag_msg}"
        )
        return avg

    def fit(
        self,
        *,
        train_loader: Iterable[dict[str, Any]],
        val_loader: Iterable[dict[str, Any]] | None = None,
        run_name: str = "default_run",
        start_epoch: int = 1,
    ) -> dict[str, float]:
        final_train: dict[str, float] = {}
        final_val: dict[str, float] = {}
        current_epoch = max(start_epoch, 1)

        ckpt_dir = Path(self.config.checkpoint_dir) / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            for epoch in range(max(start_epoch, 1), self.config.epochs + 1):
                current_epoch = epoch
                final_train = self.train_one_epoch(train_loader, epoch)

                if val_loader is not None:
                    final_val = self.validate(val_loader, epoch)
                    selected_metric = self._resolve_best_metric_from_val(final_val)

                    if self._is_better_metric(selected_metric):
                        self.best_val_loss = selected_metric
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            global_step=self.global_step,
                            best_metric=selected_metric,
                            extra={
                                "stage": self.stage.name,
                                "proxy_enabled": self.proxy_scorer is not None,
                                "best_metric_name": self.config.best_metric_name,
                                "best_metric_mode": self._best_metric_mode,
                                "proxy_total_score": final_val.get("proxy_total_score"),
                            },
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
                    extra={
                        "stage": self.stage.name,
                        "proxy_enabled": self.proxy_scorer is not None,
                        "best_metric_name": self.config.best_metric_name,
                        "best_metric_mode": self._best_metric_mode,
                        "proxy_total_score": final_val.get("proxy_total_score") if final_val else None,
                    },
                )
        except KeyboardInterrupt:
            save_checkpoint(
                ckpt_dir / "interrupted.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=current_epoch,
                global_step=self.global_step,
                best_metric=self.best_val_loss,
                extra={
                    "stage": self.stage.name,
                    "interrupted": True,
                    "proxy_enabled": self.proxy_scorer is not None,
                    "best_metric_name": self.config.best_metric_name,
                    "best_metric_mode": self._best_metric_mode,
                },
            )
            print(f"[warn] Training interrupted. Saved checkpoint: {ckpt_dir / 'interrupted.pt'}")
            raise

        return {
            "train_total": final_train.get("total", float("nan")),
            "val_total": final_val.get("total", float("nan")) if final_val else float("nan"),
            "best_val_total": float(self.best_val_loss) if self.best_val_loss is not None else float("nan"),
            "best_metric_name": str(self.config.best_metric_name),
            "best_metric_mode": self._best_metric_mode,
            "best_metric_value": float(self.best_val_loss) if self.best_val_loss is not None else float("nan"),
            "val_proxy_total": final_val.get("proxy_total_score", float("nan")) if final_val else float("nan"),
        }
