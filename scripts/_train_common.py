from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.composite import CompositeLoss, CompositeLossConfig
from src.models.hybrid_model import HybridLensCorrectionModel
from src.train.checkpointing import load_checkpoint
from src.train.config_loader import (
    dump_loaded_configs,
    load_loss_config,
    load_model_config,
    load_train_config,
)
from src.train.engine import TrainerEngine
from src.train.optim import create_optimizer, create_scheduler
from src.train.stage_configs import get_stage_toggles
from src.train.warp_backends import MockWarpBackend, Person1GeometryWarpBackend


def build_synthetic_loader(
    *,
    num_steps: int,
    batch_size: int,
    height: int,
    width: int,
) -> list[dict[str, torch.Tensor]]:
    loader: list[dict[str, torch.Tensor]] = []
    for _ in range(max(num_steps, 1)):
        inp = torch.rand(batch_size, 3, height, width, dtype=torch.float32)
        tgt = torch.rand(batch_size, 3, height, width, dtype=torch.float32)
        loader.append({"input_image": inp, "target_image": tgt})
    return loader


def _optional_path(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value if value else None


def _require_existing_file(path: str, *, label: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    if not p.is_file():
        raise ValueError(f"{label} must be a file: {p}")


def _validate_stage_loss_alignment(*, train_stage: str, loss_cfg: CompositeLossConfig) -> None:
    if loss_cfg.stage != train_stage:
        raise ValueError(f"Stage mismatch: train_config.stage='{train_stage}' vs loss_config.stage='{loss_cfg.stage}'")
    if train_stage == "stage1_param_only":
        if (
            loss_cfg.flow_tv_weight > 0.0
            or loss_cfg.flow_mag_weight > 0.0
            or loss_cfg.flow_curv_weight > 0.0
            or loss_cfg.jacobian_weight > 0.0
        ):
            raise ValueError("stage1_param_only requires zero flow/jacobian regularizer weights in loss config.")


def _resolve_external_loaders(
    *,
    module_name: str,
    function_name: str,
    stage: str,
) -> tuple[Iterable[dict[str, Any]], Iterable[dict[str, Any]] | None]:
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise ValueError(f"Loader function is not callable: {module_name}.{function_name}")

    out = fn(stage=stage)
    if not isinstance(out, tuple) or len(out) != 2:
        raise ValueError(
            f"{module_name}.{function_name} must return (train_loader, val_loader), got {type(out)}"
        )
    return out[0], out[1]


def _pick_warp_backend(name: str):
    lname = name.lower()
    if lname == "person1":
        return Person1GeometryWarpBackend()
    if lname == "mock":
        return MockWarpBackend()
    raise ValueError(f"Unsupported warp_backend: {name}")


def _steps_per_epoch(loader: Iterable[Any], fallback: int) -> int:
    try:
        n = len(loader)  # type: ignore[arg-type]
        if n > 0:
            return int(n)
    except Exception:
        pass
    return max(int(fallback), 1)


def run_training_from_configs(
    *,
    model_config_path: str,
    loss_config_path: str,
    train_config_path: str,
    use_synthetic: bool,
    loader_module: str | None,
    loader_fn: str,
    resume_from: str | None = None,
    init_from: str | None = None,
    validate_only: bool = False,
    run_name_override: str | None = None,
    warp_backend_override: str | None = None,
) -> dict[str, float]:
    _require_existing_file(model_config_path, label="Model config")
    _require_existing_file(loss_config_path, label="Loss config")
    _require_existing_file(train_config_path, label="Train config")

    resume_path = _optional_path(resume_from)
    init_path = _optional_path(init_from)
    if resume_path is not None and init_path is not None:
        print("[info] Both --resume-from and --init-from provided; ignoring --init-from and resuming full state.")
        init_path = None
    if resume_path is not None:
        _require_existing_file(resume_path, label="Resume checkpoint")
    if init_path is not None:
        _require_existing_file(init_path, label="Init checkpoint")

    model_cfg, bounds = load_model_config(model_config_path)
    loss_cfg = load_loss_config(loss_config_path)
    engine_cfg, optim_cfg, sched_cfg, extra = load_train_config(train_config_path)

    run_name = run_name_override or extra["run_name"]
    stage_name = extra["stage"]
    _validate_stage_loss_alignment(train_stage=stage_name, loss_cfg=loss_cfg)

    model = HybridLensCorrectionModel(config=model_cfg, param_bounds=bounds)
    loss_fn = CompositeLoss(loss_cfg)
    stage = get_stage_toggles(stage_name)

    backend_name = warp_backend_override or extra["warp_backend"]
    warp_backend = _pick_warp_backend(backend_name)

    if use_synthetic:
        train_loader = build_synthetic_loader(
            num_steps=extra["synthetic_train_steps"],
            batch_size=extra["synthetic_batch_size"],
            height=extra["synthetic_height"],
            width=extra["synthetic_width"],
        )
        val_loader = build_synthetic_loader(
            num_steps=extra["synthetic_val_steps"],
            batch_size=extra["synthetic_batch_size"],
            height=extra["synthetic_height"],
            width=extra["synthetic_width"],
        )
    else:
        if loader_module is None:
            raise ValueError("Real-data training requires --loader-module (or pass --use-synthetic).")
        train_loader, val_loader = _resolve_external_loaders(
            module_name=loader_module,
            function_name=loader_fn,
            stage=stage_name,
        )

    optimizer = create_optimizer(model, optim_cfg)

    steps_per_epoch = _steps_per_epoch(train_loader, fallback=extra["synthetic_train_steps"])
    total_steps = max(steps_per_epoch * max(engine_cfg.epochs, 1), 1)
    sched_bundle = create_scheduler(
        optimizer,
        sched_cfg,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )

    trainer = TrainerEngine(
        model=model,
        loss_fn=loss_fn,
        stage=stage,
        warp_backend=warp_backend,
        optimizer=optimizer,
        scheduler=sched_bundle.scheduler,
        scheduler_step_interval=sched_bundle.step_interval,
        config=engine_cfg,
    )

    start_epoch = 1
    if resume_path is not None:
        try:
            meta = load_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=sched_bundle.scheduler,
                scaler=trainer.scaler,
                map_location=trainer.device,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to resume from checkpoint '{resume_path}'. "
                "Ensure model/loss/train configs match the original run."
            ) from exc
        trainer.global_step = int(meta.get("global_step", 0))
        best_metric = meta.get("best_metric")
        trainer.best_val_loss = float(best_metric) if best_metric is not None else None
        start_epoch = int(meta.get("epoch", 0)) + 1
        print(f"[info] Resumed training from {resume_path} at epoch={start_epoch}.")
    elif init_path is not None:
        try:
            _ = load_checkpoint(
                init_path,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                map_location=trainer.device,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to initialize from checkpoint '{init_path}'. "
                "Checkpoint architecture must match current model config."
            ) from exc
        print(f"[info] Loaded initialization weights from {init_path}.")

    if validate_only:
        if val_loader is None:
            raise ValueError("Validation-only mode requires a validation loader.")
        val_metrics = trainer.validate(val_loader, epoch=max(start_epoch - 1, 0))
        return {
            "train_total": float("nan"),
            "val_total": float(val_metrics.get("total", float("nan"))),
            "best_val_total": float(trainer.best_val_loss) if trainer.best_val_loss is not None else float("nan"),
            "val_proxy_total": float(val_metrics.get("proxy_total_score", float("nan"))),
        }

    metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=run_name,
        start_epoch=start_epoch,
    )

    cfg_dump = dump_loaded_configs(
        model_cfg=model_cfg,
        bounds=bounds,
        loss_cfg=loss_cfg,
        engine_cfg=engine_cfg,
        optim_cfg=optim_cfg,
        sched_cfg=sched_cfg,
    )
    cfg_dump["runtime"] = {
        "run_name": run_name,
        "stage": stage_name,
        "warp_backend": backend_name,
        "resume_from": resume_path,
        "init_from": init_path,
        "validate_only": bool(validate_only),
        "use_synthetic": bool(use_synthetic),
        "loader_module": loader_module,
        "loader_fn": loader_fn,
    }

    out_dir = Path(engine_cfg.checkpoint_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")

    return metrics


def build_parser(
    default_model_cfg: str,
    default_loss_cfg: str,
    default_train_cfg: str,
    *,
    default_init_from: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage training with Person 2 stack.")
    parser.add_argument("--model-config", default=default_model_cfg)
    parser.add_argument("--loss-config", default=default_loss_cfg)
    parser.add_argument("--train-config", default=default_train_cfg)
    parser.add_argument("--run-name", default=None, help="Optional run-name override.")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic loader instead of external data module.",
    )
    parser.add_argument(
        "--loader-module",
        default="src.data.real_loader",
        help="External data module that provides (train_loader, val_loader).",
    )
    parser.add_argument(
        "--loader-fn",
        default="build_train_val_loaders",
        help="Function in loader-module returning (train_loader, val_loader).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume full optimizer/scheduler/scaler state from checkpoint.",
    )
    parser.add_argument(
        "--init-from",
        default=default_init_from,
        help="Initialize model weights from checkpoint (model-only load). Pass empty string to disable.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation loop only (no training updates).",
    )
    parser.add_argument(
        "--warp-backend",
        default=None,
        help="Optional warp backend override (person1|mock).",
    )
    return parser
