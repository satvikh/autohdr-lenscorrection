from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.losses.composite import CompositeLossConfig
from src.models.heads_parametric import ParametricBounds
from src.models.hybrid_model import HybridModelConfig
from src.train.engine import EngineConfig
from src.train.optim import OptimConfig, SchedulerConfig

VALID_STAGES = {"stage1_param_only", "stage2_hybrid", "stage3_finetune"}
VALID_WARP_BACKENDS = {"person1", "mock"}
VALID_SCHEDULERS = {"none", "cosine", "onecycle"}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML at {path} must be a mapping")
    return obj


def _read_range(bounds_raw: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = bounds_raw.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"param_bounds.{key} must be [min, max]")
    lo = float(raw[0])
    hi = float(raw[1])
    if not (hi > lo):
        raise ValueError(f"param_bounds.{key} must satisfy max > min, got ({lo}, {hi})")
    return (lo, hi)


def _assert_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _assert_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _validate_stage(name: str, *, field_name: str) -> str:
    stage = str(name).strip().lower()
    if stage not in VALID_STAGES:
        valid = ", ".join(sorted(VALID_STAGES))
        raise ValueError(f"{field_name} must be one of [{valid}], got '{name}'")
    return stage


def load_model_config(path: str | Path) -> tuple[HybridModelConfig, ParametricBounds]:
    raw = _load_yaml(path)

    bounds_raw = raw.get("param_bounds", {})
    if not isinstance(bounds_raw, dict):
        raise ValueError("param_bounds must be a mapping")

    bounds = ParametricBounds(
        k1=_read_range(bounds_raw, "k1", (-0.6, 0.6)),
        k2=_read_range(bounds_raw, "k2", (-0.3, 0.3)),
        k3=_read_range(bounds_raw, "k3", (-0.15, 0.15)),
        p1=_read_range(bounds_raw, "p1", (-0.03, 0.03)),
        p2=_read_range(bounds_raw, "p2", (-0.03, 0.03)),
        dcx=_read_range(bounds_raw, "dcx", (-0.08, 0.08)),
        dcy=_read_range(bounds_raw, "dcy", (-0.08, 0.08)),
        scale=_read_range(bounds_raw, "scale", (0.90, 1.20)),
        aspect=_read_range(bounds_raw, "aspect", (0.97, 1.03)),
    )

    cfg = HybridModelConfig(
        backbone_name=str(raw.get("backbone_name", "resnet34")),
        pretrained_backbone=bool(raw.get("pretrained_backbone", False)),
        use_coord_channels=bool(raw.get("use_coord_channels", True)),
        include_aspect=bool(raw.get("include_aspect", False)),
        param_hidden_dim=int(raw.get("param_hidden_dim", 256)),
        residual_hidden_dim=int(raw.get("residual_hidden_dim", 128)),
        residual_max_disp=float(raw.get("residual_max_disp", 8.0)),
        return_pred_image=bool(raw.get("return_pred_image", False)),
    )
    _assert_positive("model.param_hidden_dim", float(cfg.param_hidden_dim))
    _assert_positive("model.residual_hidden_dim", float(cfg.residual_hidden_dim))
    _assert_positive("model.residual_max_disp", cfg.residual_max_disp)
    return cfg, bounds


def load_loss_config(path: str | Path) -> CompositeLossConfig:
    raw = _load_yaml(path)
    scales = tuple(float(x) for x in raw.get("multiscale_scales", [1.0, 0.5]))
    if len(scales) == 0:
        raise ValueError("multiscale_scales must contain at least one scale")
    for i, s in enumerate(scales):
        _assert_positive(f"multiscale_scales[{i}]", s)

    cfg = CompositeLossConfig(
        stage=_validate_stage(str(raw.get("stage", "stage1_param_only")), field_name="loss.stage"),
        use_charbonnier=bool(raw.get("use_charbonnier", True)),
        charbonnier_eps=float(raw.get("charbonnier_eps", 1e-3)),
        multiscale_scales=scales,
        pixel_weight=float(raw.get("pixel_weight", 0.10)),
        ssim_weight=float(raw.get("ssim_weight", 0.15)),
        edge_weight=float(raw.get("edge_weight", 0.40)),
        grad_orient_weight=float(raw.get("grad_orient_weight", 0.18)),
        flow_tv_weight=float(raw.get("flow_tv_weight", 0.0)),
        flow_mag_weight=float(raw.get("flow_mag_weight", 0.0)),
        flow_curv_weight=float(raw.get("flow_curv_weight", 0.0)),
        jacobian_weight=float(raw.get("jacobian_weight", 0.0)),
        jacobian_margin=float(raw.get("jacobian_margin", 0.0)),
    )
    _assert_positive("loss.charbonnier_eps", cfg.charbonnier_eps)
    for name in (
        "pixel_weight",
        "ssim_weight",
        "edge_weight",
        "grad_orient_weight",
        "flow_tv_weight",
        "flow_mag_weight",
        "flow_curv_weight",
        "jacobian_weight",
        "jacobian_margin",
    ):
        _assert_non_negative(f"loss.{name}", float(getattr(cfg, name)))
    return cfg


def load_train_config(path: str | Path) -> tuple[EngineConfig, OptimConfig, SchedulerConfig, dict[str, Any]]:
    raw = _load_yaml(path)
    stage_name = _validate_stage(str(raw.get("stage", "stage1_param_only")), field_name="train.stage")
    run_name = str(raw.get("run_name", "default_run")).strip()
    if not run_name:
        raise ValueError("train.run_name must be a non-empty string")

    warp_backend = str(raw.get("warp_backend", "person1")).strip().lower()
    if warp_backend not in VALID_WARP_BACKENDS:
        valid = ", ".join(sorted(VALID_WARP_BACKENDS))
        raise ValueError(f"train.warp_backend must be one of [{valid}], got '{warp_backend}'")

    best_metric_name = str(raw.get("best_metric_name", "total"))
    best_metric_mode = str(raw.get("best_metric_mode", "min")).strip().lower()
    if best_metric_name.strip() == "":
        raise ValueError("train.best_metric_name must be a non-empty string")
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(f"train.best_metric_mode must be 'min' or 'max', got '{best_metric_mode}'")

    engine = EngineConfig(
        epochs=int(raw.get("epochs", 1)),
        amp_enabled=bool(raw.get("amp_enabled", False)),
        grad_clip_norm=float(raw.get("grad_clip_norm", 1.0)) if raw.get("grad_clip_norm") is not None else None,
        log_interval=int(raw.get("log_interval", 10)),
        device=str(raw.get("device", "cpu")),
        max_steps_per_epoch=int(raw.get("max_steps_per_epoch")) if raw.get("max_steps_per_epoch") is not None else None,
        max_val_steps=int(raw.get("max_val_steps")) if raw.get("max_val_steps") is not None else None,
        checkpoint_dir=str(raw.get("checkpoint_dir", "outputs/runs")),
        proxy_enabled=bool(raw.get("proxy_enabled", False)),
        proxy_module_path=str(raw.get("proxy_module_path")) if raw.get("proxy_module_path") is not None else None,
        proxy_function_name=str(raw.get("proxy_function_name", "compute_proxy_score")),
        proxy_config=raw.get("proxy_config"),
        debug_dump_dir=str(raw.get("debug_dump_dir")) if raw.get("debug_dump_dir") is not None else None,
        debug_dump_max_images=int(raw.get("debug_dump_max_images", 0)),
        fail_on_nonfinite_loss=bool(raw.get("fail_on_nonfinite_loss", True)),
        param_saturation_warn_threshold=float(raw.get("param_saturation_warn_threshold", 0.50)),
        residual_warn_abs_max_px=float(raw.get("residual_warn_abs_max_px", 20.0)),
        best_metric_name=best_metric_name,
        best_metric_mode=best_metric_mode,
    )
    if engine.epochs < 1:
        raise ValueError(f"train.epochs must be >= 1, got {engine.epochs}")
    if engine.log_interval < 0:
        raise ValueError(f"train.log_interval must be >= 0, got {engine.log_interval}")
    if engine.max_steps_per_epoch is not None and engine.max_steps_per_epoch <= 0:
        raise ValueError("train.max_steps_per_epoch must be > 0 when set")
    if engine.max_val_steps is not None and engine.max_val_steps <= 0:
        raise ValueError("train.max_val_steps must be > 0 when set")
    if engine.grad_clip_norm is not None:
        _assert_non_negative("train.grad_clip_norm", float(engine.grad_clip_norm))
    _assert_non_negative("train.param_saturation_warn_threshold", engine.param_saturation_warn_threshold)
    _assert_non_negative("train.residual_warn_abs_max_px", engine.residual_warn_abs_max_px)

    optim_raw = raw.get("optimizer", {})
    if not isinstance(optim_raw, dict):
        raise ValueError("optimizer must be a mapping")
    betas_raw = optim_raw.get("betas", [0.9, 0.999])
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise ValueError("optimizer.betas must have exactly 2 values")

    optim = OptimConfig(
        lr=float(optim_raw.get("lr", 1e-4)),
        weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
        betas=(float(betas_raw[0]), float(betas_raw[1])),
    )
    _assert_positive("optimizer.lr", optim.lr)
    _assert_non_negative("optimizer.weight_decay", optim.weight_decay)
    if not (0.0 < optim.betas[0] < 1.0 and 0.0 < optim.betas[1] < 1.0):
        raise ValueError(f"optimizer.betas must each be in (0,1), got {optim.betas}")

    sched_raw = raw.get("scheduler", {})
    if not isinstance(sched_raw, dict):
        raise ValueError("scheduler must be a mapping")
    sched = SchedulerConfig(
        name=str(sched_raw.get("name", "none")),
        min_lr=float(sched_raw.get("min_lr", 1e-6)),
        pct_start=float(sched_raw.get("pct_start", 0.3)),
        div_factor=float(sched_raw.get("div_factor", 25.0)),
        final_div_factor=float(sched_raw.get("final_div_factor", 1e4)),
    )
    sched_name = sched.name.lower().strip()
    if sched_name not in VALID_SCHEDULERS:
        valid = ", ".join(sorted(VALID_SCHEDULERS))
        raise ValueError(f"scheduler.name must be one of [{valid}], got '{sched.name}'")
    _assert_non_negative("scheduler.min_lr", sched.min_lr)
    _assert_positive("scheduler.div_factor", sched.div_factor)
    _assert_positive("scheduler.final_div_factor", sched.final_div_factor)
    _assert_positive("scheduler.pct_start", sched.pct_start)
    if sched.pct_start >= 1.0:
        raise ValueError(f"scheduler.pct_start must be < 1.0, got {sched.pct_start}")

    extra = {
        "run_name": run_name,
        "stage": stage_name,
        "warp_backend": warp_backend,
        "synthetic_train_steps": int(raw.get("synthetic_train_steps", 8)),
        "synthetic_val_steps": int(raw.get("synthetic_val_steps", 2)),
        "synthetic_batch_size": int(raw.get("synthetic_batch_size", 2)),
        "synthetic_height": int(raw.get("synthetic_height", 128)),
        "synthetic_width": int(raw.get("synthetic_width", 128)),
    }
    for key in ("synthetic_train_steps", "synthetic_val_steps", "synthetic_batch_size", "synthetic_height", "synthetic_width"):
        if int(extra[key]) <= 0:
            raise ValueError(f"train.{key} must be > 0, got {extra[key]}")

    return engine, optim, sched, extra


def dump_loaded_configs(
    *,
    model_cfg: HybridModelConfig,
    bounds: ParametricBounds,
    loss_cfg: CompositeLossConfig,
    engine_cfg: EngineConfig,
    optim_cfg: OptimConfig,
    sched_cfg: SchedulerConfig,
) -> dict[str, Any]:
    return {
        "model": asdict(model_cfg),
        "bounds": asdict(bounds),
        "loss": asdict(loss_cfg),
        "engine": asdict(engine_cfg),
        "optimizer": asdict(optim_cfg),
        "scheduler": asdict(sched_cfg),
    }
