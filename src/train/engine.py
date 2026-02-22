from __future__ import annotations

from dataclasses import dataclass
import hashlib
import platform
from pathlib import Path
import time
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

    # Debug/instrumentation controls (all disabled by default).
    debug_instrumentation: bool = False
    debug_metric_precision: int = 4
    debug_log_sample_ids: bool = False
    debug_param_update_interval: int = 0
    debug_perf_enabled: bool = False
    debug_perf_interval: int = 0
    debug_probe_enabled: bool = False
    debug_probe_max_samples: int = 8
    debug_residual_inactive_threshold_px: float = 1e-5
    debug_residual_inactive_patience: int = 20
    debug_zero_grad_threshold: float = 1e-12
    debug_zero_param_delta_threshold: float = 1e-12


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
        self._residual_inactive_steps = 0
        self._train_ids_logged_epochs: set[int] = set()
        self._val_ids_logged_epochs: set[int] = set()
        self._last_probe_pred_checksum: str | None = None
        self._last_probe_params_checksum: str | None = None
        self._last_train_param_delta: float = 0.0
        mode = str(self.config.best_metric_mode).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError(f"best_metric_mode must be 'min' or 'max', got: {self.config.best_metric_mode}")
        self._best_metric_mode = mode

        if self.config.debug_instrumentation:
            self._log_runtime_header()

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

    def _maybe_step_scheduler_batch(self, *, optimizer_stepped: bool) -> None:
        if self.scheduler is None or self.scheduler_step_interval != "batch":
            return
        if not optimizer_stepped:
            return
        self.scheduler.step()

    def _maybe_step_scheduler_epoch(self) -> None:
        if self.scheduler is not None and self.scheduler_step_interval == "epoch":
            self.scheduler.step()

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _fmt(self, value: float) -> str:
        precision = max(int(self.config.debug_metric_precision), 1)
        return f"{float(value):.{precision}f}"

    def _log_runtime_header(self) -> None:
        n_backbone = 0
        n_param = 0
        n_residual = 0
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            n = int(p.numel())
            if name.startswith("backbone."):
                n_backbone += n
            elif name.startswith("param_head."):
                n_param += n
            elif name.startswith("residual_head."):
                n_residual += n

        print(
            "[debug] runtime "
            f"platform={platform.system()} "
            f"device={self.device} "
            f"cuda_available={torch.cuda.is_available()} "
            f"amp_enabled={self.config.amp_enabled} "
            f"stage={self.stage.name} "
            f"scheduler_interval={self.scheduler_step_interval}"
        )
        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            print(f"[debug] cuda_device index={idx} name={torch.cuda.get_device_name(idx)}")
        print(
            "[debug] param_groups "
            f"backbone={n_backbone} "
            f"param_head={n_param} "
            f"residual_head={n_residual}"
        )
        if self.stage.use_residual and n_residual == 0:
            print("[warn] Stage uses residual branch but residual head has zero trainable parameters.")

    @staticmethod
    def _extract_sample_ids(batch: dict[str, Any], max_items: int = 6) -> list[str]:
        ids = batch.get("image_id")
        if ids is None:
            return []
        if isinstance(ids, str):
            return [ids]
        if isinstance(ids, (list, tuple)):
            out: list[str] = []
            for item in ids[: max_items]:
                out.append(str(item))
            return out
        return [str(ids)]

    @staticmethod
    def _tensor_checksum(value: Tensor) -> str:
        arr = (
            value.detach()
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
            .numpy()
            .tobytes()
        )
        return hashlib.sha256(arr).hexdigest()[:16]

    @staticmethod
    def _avg_or_nan(values: list[float]) -> float:
        if not values:
            return float("nan")
        return float(sum(values) / len(values))

    def _collect_base_metrics(self, out_components: dict[str, Tensor], diagnostics: dict[str, float]) -> dict[str, float]:
        metrics = tensor_dict_to_float(out_components)
        metrics.update(diagnostics)
        metrics["lr"] = self._current_lr()
        return metrics

    def _log_loader_debug_once(self, loader: Iterable[dict[str, Any]], *, split: str) -> None:
        if not self.config.debug_instrumentation:
            return
        nw = getattr(loader, "num_workers", None)
        bs = getattr(loader, "batch_size", None)
        if nw is None and bs is None:
            return
        print(f"[debug] {split}_loader batch_size={bs} num_workers={nw}")
        if platform.system().lower().startswith("win") and isinstance(nw, int) and nw > 0:
            print("[warn] Windows detected with num_workers>0. Consider num_workers=0 for stability in debug runs.")

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

    def _maybe_warn_zero_grad_or_delta(self, metrics: dict[str, float], *, epoch: int, step_idx: int) -> None:
        if not self.config.debug_instrumentation:
            return
        grad_thr = float(self.config.debug_zero_grad_threshold)
        delta_thr = float(self.config.debug_zero_param_delta_threshold)
        for group in ("backbone", "param_head", "residual_head"):
            nonfinite = float(metrics.get(f"grad_nonfinite_count_{group}", 0.0))
            if nonfinite > 0:
                key = f"nonfinite_grad:{group}:{epoch}"
                if key not in self._warned_once:
                    self._warned_once.add(key)
                    print(
                        f"[warn] Non-finite gradients detected for {group} at epoch={epoch}, step={step_idx}: "
                        f"count={nonfinite:.0f}"
                    )
            g = float(metrics.get(f"grad_norm_{group}", 0.0))
            d = float(metrics.get(f"param_delta_{group}", 0.0))
            if g <= grad_thr:
                key = f"zero_grad:{group}:{epoch}"
                if key not in self._warned_once:
                    self._warned_once.add(key)
                    print(
                        f"[warn] Near-zero gradient norm for {group} at epoch={epoch}, step={step_idx}: "
                        f"{g:.3e} <= {grad_thr:.3e}"
                    )
            if d <= delta_thr:
                key = f"zero_delta:{group}:{epoch}"
                if key not in self._warned_once:
                    self._warned_once.add(key)
                    print(
                        f"[warn] Near-zero parameter delta for {group} at epoch={epoch}, step={step_idx}: "
                        f"{d:.3e} <= {delta_thr:.3e}"
                    )

    def _track_residual_activity(self, metrics: dict[str, float], *, epoch: int, step_idx: int) -> None:
        if not self.stage.use_residual:
            return
        threshold = float(self.config.debug_residual_inactive_threshold_px)
        patience = max(int(self.config.debug_residual_inactive_patience), 1)
        residual_max = float(metrics.get("residual_lowres_abs_max_px", 0.0))
        if residual_max < threshold:
            self._residual_inactive_steps += 1
        else:
            self._residual_inactive_steps = 0

        if self._residual_inactive_steps < patience:
            return
        key = f"residual_inactive:{epoch}:{step_idx // patience}"
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        print(
            f"[warn] Residual branch appears inactive at epoch={epoch}, step={step_idx}: "
            f"residual_lowres_abs_max_px={residual_max:.3e} < {threshold:.3e} "
            f"for {self._residual_inactive_steps} consecutive steps; "
            f"grad_norm_residual_head={metrics.get('grad_norm_residual_head', float('nan')):.3e}; "
            f"param_delta_residual_head={metrics.get('param_delta_residual_head', float('nan')):.3e}"
        )

    def _metric_summary(self, metrics: dict[str, float], keys: list[str]) -> str:
        parts: list[str] = []
        for k in keys:
            if k in metrics:
                parts.append(f"{k}={self._fmt(metrics[k])}")
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
        raw_candidate = val_metrics.get(key, float("nan"))
        try:
            candidate = float(raw_candidate)  # type: ignore[arg-type]
        except Exception:
            candidate = float("nan")
        if torch.isfinite(torch.tensor(candidate)):
            return candidate
        # Backward-compatible fallback keeps historical behavior.
        try:
            fallback = float(val_metrics.get("total", float("nan")))
        except Exception:
            fallback = float("nan")
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
        self.model.train()
        self._log_loader_debug_once(train_loader, split="train")
        if self.config.debug_instrumentation:
            print(f"[debug] train_mode epoch={epoch} model.training={self.model.training}")
        if self.device.type == "cuda" and self.config.debug_perf_enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
        last_step_end = time.perf_counter()

        for step_idx, batch in enumerate(train_loader, start=1):
            if self.config.max_steps_per_epoch is not None and step_idx > self.config.max_steps_per_epoch:
                break

            loop_start = time.perf_counter()
            data_time_ms = (loop_start - last_step_end) * 1000.0
            if self.config.debug_log_sample_ids and epoch not in self._train_ids_logged_epochs:
                ids = self._extract_sample_ids(batch)
                print(f"[debug] train_sample_ids_preview epoch={epoch} ids={ids}")
                self._train_ids_logged_epochs.add(epoch)

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
                debug_instrumentation=bool(self.config.debug_instrumentation),
            )
            self.global_step += 1

            metrics = self._collect_base_metrics(out.components, out.diagnostics)
            metrics["data_time_ms"] = float(data_time_ms)
            step_end = time.perf_counter()
            batch_time_ms = (step_end - loop_start) * 1000.0
            metrics["batch_time_ms"] = float(batch_time_ms)
            batch_size = 0
            if isinstance(batch.get("input_image"), torch.Tensor):
                batch_size = int(batch["input_image"].shape[0])
            if batch_size > 0:
                metrics["samples_per_sec"] = float(batch_size / max(batch_time_ms / 1000.0, 1e-9))
            if self.device.type == "cuda" and self.config.debug_perf_enabled:
                metrics["cuda_mem_allocated_mb"] = float(torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0))
                metrics["cuda_mem_reserved_mb"] = float(torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0))
                metrics["max_cuda_mem_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
                )
            if self.config.debug_instrumentation:
                grad_thr = float(self.config.debug_zero_grad_threshold)
                delta_thr = float(self.config.debug_zero_param_delta_threshold)
                metrics["residual_head_grad_is_zero"] = (
                    1.0 if float(metrics.get("grad_norm_residual_head", 0.0)) <= grad_thr else 0.0
                )
                metrics["residual_head_param_delta_is_zero"] = (
                    1.0 if float(metrics.get("param_delta_residual_head", 0.0)) <= delta_thr else 0.0
                )

            self._check_finite_train_total(float(metrics.get("total", float("nan"))), epoch=epoch, step_idx=step_idx)
            self._maybe_warn_saturation(metrics, epoch=epoch, step_idx=step_idx)
            self._maybe_warn_residual(metrics, epoch=epoch, step_idx=step_idx)
            self._maybe_warn_zero_grad_or_delta(metrics, epoch=epoch, step_idx=step_idx)
            self._track_residual_activity(metrics, epoch=epoch, step_idx=step_idx)
            tracker.update(metrics)

            upd_interval = int(self.config.debug_param_update_interval)
            if self.config.debug_instrumentation and upd_interval > 0 and step_idx % upd_interval == 0:
                print(
                    "[debug] update "
                    f"epoch={epoch} step={step_idx} "
                    f"grad_norm_backbone={metrics.get('grad_norm_backbone', float('nan')):.3e} "
                    f"grad_norm_param_head={metrics.get('grad_norm_param_head', float('nan')):.3e} "
                    f"grad_norm_residual_head={metrics.get('grad_norm_residual_head', float('nan')):.3e} "
                    f"param_delta_backbone={metrics.get('param_delta_backbone', float('nan')):.3e} "
                    f"param_delta_param_head={metrics.get('param_delta_param_head', float('nan')):.3e} "
                    f"param_delta_residual_head={metrics.get('param_delta_residual_head', float('nan')):.3e}"
                )

            optimizer_stepped = float(metrics.get("optim_step_skipped", 0.0)) < 0.5
            self._maybe_step_scheduler_batch(optimizer_stepped=optimizer_stepped)
            last_step_end = step_end

            if self.config.log_interval > 0 and step_idx % self.config.log_interval == 0:
                avg = tracker.averages()
                loss_msg = self._metric_summary(
                    avg,
                    [
                        "total",
                        "pixel",
                        "ssim",
                        "edge",
                        "grad_orient",
                        "flow_tv",
                        "flow_mag",
                        "jacobian",
                        "flow_tv_weighted",
                        "flow_mag_weighted",
                        "jacobian_weighted",
                    ],
                )
                diag_msg = self._metric_summary(
                    avg,
                    [
                        "params_raw_abs_mean",
                        "param_sat_frac_max",
                        "grad_norm_backbone",
                        "grad_norm_param_head",
                        "grad_norm_residual_head",
                        "grad_nonfinite_count_backbone",
                        "grad_nonfinite_count_param_head",
                        "grad_nonfinite_count_residual_head",
                        "param_delta_backbone",
                        "param_delta_param_head",
                        "param_delta_residual_head",
                        "residual_head_grad_is_zero",
                        "residual_head_param_delta_is_zero",
                        "optim_step_skipped",
                        "amp_scale_before",
                        "amp_scale_after",
                        "amp_scale_ratio",
                        "residual_lowres_abs_mean_px",
                        "residual_lowres_abs_max_px",
                        "residual_contrib_px_mean",
                        "residual_to_param_disp_ratio_mean",
                        "input_min",
                        "input_max",
                        "target_min",
                        "target_max",
                        "pred_min",
                        "pred_max",
                        "warp_oob_ratio",
                        "warp_negative_det_pct",
                        "warp_safety_safe",
                    ],
                )
                perf_keys: list[str] = []
                perf_interval = int(self.config.debug_perf_interval) if self.config.debug_perf_interval > 0 else self.config.log_interval
                if self.config.debug_perf_enabled and perf_interval > 0 and step_idx % perf_interval == 0:
                    perf_keys = [
                        "data_time_ms",
                        "timing_forward_ms",
                        "timing_backward_ms",
                        "timing_optim_step_ms",
                        "batch_time_ms",
                        "samples_per_sec",
                        "cuda_mem_allocated_mb",
                        "cuda_mem_reserved_mb",
                        "max_cuda_mem_allocated_mb",
                    ]
                perf_msg = self._metric_summary(avg, perf_keys) if perf_keys else ""
                print(
                    f"[train] epoch={epoch} step={step_idx} "
                    f"{loss_msg} "
                    f"lr={avg.get('lr', float('nan')):.6e} "
                    f"{diag_msg} "
                    f"{perf_msg}"
                )

        self._maybe_step_scheduler_epoch()
        return tracker.averages()

    def validate(self, val_loader: Iterable[dict[str, Any]], epoch: int) -> dict[str, float]:
        tracker = RunningAverage()
        self._log_loader_debug_once(val_loader, split="val")
        if self.config.debug_instrumentation:
            print(f"[debug] eval_mode epoch={epoch} model.training_pre_eval={self.model.training}")

        val_batches_processed = 0
        val_samples_processed = 0
        val_proxy_predictions = 0

        probe_enabled = bool(self.config.debug_probe_enabled)
        probe_limit = max(int(self.config.debug_probe_max_samples), 0)
        probe_pred: list[Tensor] = []
        probe_target: list[Tensor] = []
        probe_params: list[Tensor] = []
        probe_final_grid: list[Tensor] = []
        probe_component_sums: dict[str, float] = {"pixel": 0.0, "ssim": 0.0, "edge": 0.0}
        probe_component_count = 0
        probe_collected = 0

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
            if self.config.debug_instrumentation and self.model.training:
                print(f"[warn] Model remained in train mode during validation at epoch={epoch}, step={step_idx}.")
            val_batches_processed += 1

            metrics = self._collect_base_metrics(out.components, out.diagnostics)

            if isinstance(batch.get("target_image"), torch.Tensor):
                target = batch["target_image"].to(device=out.pred_image.device, dtype=out.pred_image.dtype)
                val_samples_processed += int(target.shape[0])
                proxy_metrics = self._maybe_proxy_metrics(out.pred_image, target)
                metrics.update(proxy_metrics)
                val_proxy_predictions += int(target.shape[0])

                if isinstance(batch.get("input_image"), torch.Tensor):
                    self._maybe_debug_dump(
                        prefix=f"epoch{epoch}_step{step_idx}",
                        input_image=batch["input_image"].to(device=out.pred_image.device, dtype=out.pred_image.dtype),
                        pred_image=out.pred_image,
                        target_image=target,
                    )
                if self.config.debug_log_sample_ids and epoch not in self._val_ids_logged_epochs:
                    ids = self._extract_sample_ids(batch)
                    print(f"[debug] val_sample_ids_preview epoch={epoch} ids={ids}")
                    self._val_ids_logged_epochs.add(epoch)

                if probe_enabled and probe_limit > 0 and probe_collected < probe_limit:
                    take = min(int(target.shape[0]), probe_limit - probe_collected)
                    probe_pred.append(out.pred_image[:take].detach().cpu())
                    probe_target.append(target[:take].detach().cpu())
                    params = out.model_outputs.get("params")
                    final_grid = out.warp_outputs.get("final_grid")
                    if torch.is_tensor(params):
                        probe_params.append(params[:take].detach().cpu())
                    if torch.is_tensor(final_grid):
                        probe_final_grid.append(final_grid[:take].detach().cpu())
                    probe_collected += take
                    probe_component_sums["pixel"] += float(out.components.get("pixel", torch.tensor(0.0)).detach().item())
                    probe_component_sums["ssim"] += float(out.components.get("ssim", torch.tensor(0.0)).detach().item())
                    probe_component_sums["edge"] += float(out.components.get("edge", torch.tensor(0.0)).detach().item())
                    probe_component_count += 1

            tracker.update(metrics)

        avg = tracker.averages()
        avg["val_batches_processed"] = float(val_batches_processed)
        avg["val_samples_processed"] = float(val_samples_processed)
        avg["val_proxy_predictions_processed"] = float(val_proxy_predictions)

        if probe_enabled and probe_collected > 0:
            pred_probe = torch.cat(probe_pred, dim=0)
            target_probe = torch.cat(probe_target, dim=0)
            probe_pred_checksum = self._tensor_checksum(pred_probe)
            avg["probe_pred_checksum"] = probe_pred_checksum
            avg["probe_proxy_samples"] = float(pred_probe.shape[0])
            probe_params_checksum: str | None = None
            if probe_params:
                probe_params_checksum = self._tensor_checksum(torch.cat(probe_params, dim=0))
                avg["probe_params_checksum"] = probe_params_checksum
            if probe_final_grid:
                avg["probe_final_grid_checksum"] = self._tensor_checksum(torch.cat(probe_final_grid, dim=0))

            avg["probe_pixel"] = self._avg_or_nan([probe_component_sums["pixel"] / max(probe_component_count, 1)])
            avg["probe_ssim"] = self._avg_or_nan([probe_component_sums["ssim"] / max(probe_component_count, 1)])
            avg["probe_edge"] = self._avg_or_nan([probe_component_sums["edge"] / max(probe_component_count, 1)])

            if self.proxy_scorer is not None:
                probe_proxy = compute_proxy_metrics_for_batch(
                    scorer=self.proxy_scorer,
                    pred_batch=pred_probe,
                    target_batch=target_probe,
                    config=self.config.proxy_config,
                )
                for k, v in probe_proxy.items():
                    avg[f"probe_{k}"] = float(v)

            if self._last_probe_pred_checksum is not None and probe_pred_checksum == self._last_probe_pred_checksum:
                if self.config.debug_instrumentation:
                    print(
                        f"[warn] Probe prediction checksum unchanged at epoch={epoch} ({probe_pred_checksum}). "
                        f"Validation may be flat. last_train_param_delta={self._last_train_param_delta:.3e}"
                    )
            if (
                probe_params_checksum is not None
                and self._last_probe_params_checksum is not None
                and probe_params_checksum == self._last_probe_params_checksum
                and self.config.debug_instrumentation
            ):
                print(
                    f"[warn] Probe params checksum unchanged at epoch={epoch} ({probe_params_checksum}). "
                    "Model parameter outputs may not be moving on probe subset."
                )
            self._last_probe_pred_checksum = probe_pred_checksum
            if probe_params_checksum is not None:
                self._last_probe_params_checksum = probe_params_checksum

        loss_msg = self._metric_summary(
            avg,
            [
                "total",
                "pixel",
                "ssim",
                "edge",
                "grad_orient",
                "flow_tv",
                "flow_mag",
                "jacobian",
                "flow_tv_weighted",
                "flow_mag_weighted",
                "jacobian_weighted",
            ],
        )
        diag_msg = self._metric_summary(
            avg,
            [
                "warp_oob_ratio",
                "warp_negative_det_pct",
                "warp_safety_safe",
                "proxy_total_score",
                "proxy_edge",
                "proxy_line",
                "proxy_grad",
                "proxy_ssim",
                "proxy_mae",
                "proxy_hard_fail",
                "val_samples_processed",
                "val_proxy_predictions_processed",
                "probe_proxy_total_score",
                "probe_pixel",
                "probe_ssim",
                "probe_edge",
            ],
        )
        print(
            f"[val] epoch={epoch} {loss_msg} {diag_msg}"
        )
        if probe_enabled and probe_collected > 0:
            print(
                "[val-probe] "
                f"epoch={epoch} "
                f"probe_pred_checksum={avg.get('probe_pred_checksum')} "
                f"probe_final_grid_checksum={avg.get('probe_final_grid_checksum', 'na')} "
                f"probe_params_checksum={avg.get('probe_params_checksum', 'na')}"
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
                self._last_train_param_delta = float(
                    final_train.get("param_delta_backbone", 0.0)
                    + final_train.get("param_delta_param_head", 0.0)
                    + final_train.get("param_delta_residual_head", 0.0)
                )

                if val_loader is not None:
                    if self.config.debug_instrumentation:
                        print(
                            f"[debug] validate_start epoch={epoch} "
                            "using current in-memory model state (no checkpoint reload)."
                        )
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
