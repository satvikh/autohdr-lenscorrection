from __future__ import annotations

from pathlib import Path

import torch

from src.losses.composite import CompositeLoss, config_for_stage
from src.models.hybrid_model import HybridLensCorrectionModel, HybridModelConfig
from src.train.checkpointing import load_checkpoint, save_checkpoint
from src.train.config_loader import load_train_config
from src.train.engine import EngineConfig, TrainerEngine
from src.train.optim import OptimConfig, SchedulerConfig, create_optimizer, create_scheduler
from src.train.stage_configs import get_stage_toggles
from src.train.train_step import run_eval_step, run_train_step
from src.train.warp_backends import MockWarpBackend, Person1GeometryWarpBackend


def _dummy_batch(batch: int = 2, h: int = 64, w: int = 64) -> dict[str, torch.Tensor]:
    inp = torch.rand(batch, 3, h, w, dtype=torch.float32)
    tgt = torch.rand(batch, 3, h, w, dtype=torch.float32)
    return {"input_image": inp, "target_image": tgt}


def test_train_step_smoke_updates_parameters() -> None:
    device = torch.device("cpu")

    model = HybridLensCorrectionModel(
        config=HybridModelConfig(backbone_name="tiny", use_coord_channels=True, residual_max_disp=4.0)
    ).to(device)

    loss_fn = CompositeLoss(config_for_stage("stage2_hybrid"))
    stage = get_stage_toggles("stage2_hybrid")
    backend = MockWarpBackend()

    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))

    batch = _dummy_batch()

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    out = run_train_step(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        warp_backend=backend,
        stage=stage,
        optimizer=optimizer,
        scaler=None,
        amp_enabled=False,
        grad_clip_norm=1.0,
        device=device,
    )

    assert torch.isfinite(out.total_loss)
    assert out.total_loss.item() > 0.0
    assert "total" in out.components
    assert out.pred_image.shape == batch["input_image"].shape
    assert "param_sat_frac_max" in out.diagnostics

    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    total_delta = torch.stack([(a - b).abs().mean() for a, b in zip(after, before)]).sum().item()
    assert total_delta > 0.0

    eval_out = run_eval_step(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        warp_backend=backend,
        stage=stage,
        amp_enabled=False,
        device=device,
    )
    assert torch.isfinite(eval_out.total_loss)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny")).to(device)
    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))

    sched_bundle = create_scheduler(
        optimizer,
        SchedulerConfig(name="cosine"),
        total_steps=10,
        steps_per_epoch=5,
    )

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=sched_bundle.scheduler,
        scaler=None,
        epoch=3,
        global_step=17,
        best_metric=0.123,
        extra={"stage": "stage1_param_only"},
    )

    new_model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny")).to(device)
    new_opt = create_optimizer(new_model, OptimConfig(lr=1e-3, weight_decay=0.0))
    new_sched_bundle = create_scheduler(
        new_opt,
        SchedulerConfig(name="cosine"),
        total_steps=10,
        steps_per_epoch=5,
    )

    meta = load_checkpoint(
        ckpt_path,
        model=new_model,
        optimizer=new_opt,
        scheduler=new_sched_bundle.scheduler,
        scaler=None,
        map_location="cpu",
    )

    assert meta["epoch"] == 3
    assert meta["global_step"] == 17
    assert abs(float(meta["best_metric"]) - 0.123) < 1e-8


def test_trainer_engine_smoke(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny")).to(device)
    loss_fn = CompositeLoss(config_for_stage("stage1_param_only"))
    stage = get_stage_toggles("stage1_param_only")
    backend = MockWarpBackend()

    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))
    sched_bundle = create_scheduler(
        optimizer,
        SchedulerConfig(name="none"),
        total_steps=2,
        steps_per_epoch=2,
    )

    engine = TrainerEngine(
        model=model,
        loss_fn=loss_fn,
        stage=stage,
        warp_backend=backend,
        optimizer=optimizer,
        scheduler=sched_bundle.scheduler,
        scheduler_step_interval=sched_bundle.step_interval,
        config=EngineConfig(
            epochs=1,
            amp_enabled=False,
            grad_clip_norm=1.0,
            log_interval=1,
            device="cpu",
            max_steps_per_epoch=2,
            max_val_steps=1,
            checkpoint_dir=str(tmp_path / "runs"),
        ),
    )

    train_loader = [_dummy_batch(), _dummy_batch()]
    val_loader = [_dummy_batch()]

    metrics = engine.fit(train_loader=train_loader, val_loader=val_loader, run_name="smoke_engine")
    assert "train_total" in metrics
    assert "val_total" in metrics
    assert (tmp_path / "runs" / "smoke_engine" / "last.pt").exists()


def test_stage1_disables_residual_path() -> None:
    class SpyBackend(MockWarpBackend):
        def __init__(self) -> None:
            super().__init__()
            self.last_residual_is_none: bool | None = None

        def warp(self, image: torch.Tensor, params: torch.Tensor, residual_flow_lowres: torch.Tensor | None):
            self.last_residual_is_none = residual_flow_lowres is None
            return super().warp(image, params, residual_flow_lowres)

    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny"))
    loss_fn = CompositeLoss(config_for_stage("stage1_param_only"))
    stage = get_stage_toggles("stage1_param_only")
    backend = SpyBackend()
    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))

    _ = run_train_step(
        model=model,
        batch=_dummy_batch(),
        loss_fn=loss_fn,
        warp_backend=backend,
        stage=stage,
        optimizer=optimizer,
        scaler=None,
        amp_enabled=False,
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
    )
    assert backend.last_residual_is_none is True


def test_person1_geometry_backend_smoke() -> None:
    device = torch.device("cpu")
    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny")).to(device)
    loss_fn = CompositeLoss(config_for_stage("stage2_hybrid"))
    stage = get_stage_toggles("stage2_hybrid")
    backend = Person1GeometryWarpBackend()
    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))

    out = run_train_step(
        model=model,
        batch=_dummy_batch(),
        loss_fn=loss_fn,
        warp_backend=backend,
        stage=stage,
        optimizer=optimizer,
        scaler=None,
        amp_enabled=False,
        grad_clip_norm=1.0,
        device=device,
    )
    assert torch.isfinite(out.total_loss)
    assert "warp_oob_ratio" in out.diagnostics
    assert "warp_safety_safe" in out.diagnostics


def test_train_config_loader_smoke() -> None:
    engine, optim, sched, extra = load_train_config("configs/train/debug_smoke.yaml")
    assert engine.epochs >= 1
    assert optim.lr > 0.0
    assert isinstance(sched.name, str)
    assert extra["stage"] == "stage1_param_only"


def test_train_step_raises_on_non_finite_loss() -> None:
    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny"))
    loss_fn = CompositeLoss(config_for_stage("stage2_hybrid"))
    stage = get_stage_toggles("stage2_hybrid")
    backend = MockWarpBackend()
    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))

    batch = _dummy_batch()
    batch["target_image"][0, 0, 0, 0] = float("nan")

    try:
        _ = run_train_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            warp_backend=backend,
            stage=stage,
            optimizer=optimizer,
            scaler=None,
            amp_enabled=False,
            grad_clip_norm=1.0,
            device=torch.device("cpu"),
        )
        assert False, "Expected FloatingPointError for non-finite loss."
    except FloatingPointError:
        pass


def test_stage2_requires_final_grid_from_warp_backend() -> None:
    class NoFinalGridBackend(MockWarpBackend):
        def warp(self, image: torch.Tensor, params: torch.Tensor, residual_flow_lowres: torch.Tensor | None):
            out = super().warp(image, params, residual_flow_lowres)
            out.pop("final_grid", None)
            return out

    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny"))
    loss_fn = CompositeLoss(config_for_stage("stage2_hybrid"))
    stage = get_stage_toggles("stage2_hybrid")
    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))

    try:
        _ = run_train_step(
            model=model,
            batch=_dummy_batch(),
            loss_fn=loss_fn,
            warp_backend=NoFinalGridBackend(),
            stage=stage,
            optimizer=optimizer,
            scaler=None,
            amp_enabled=False,
            grad_clip_norm=1.0,
            device=torch.device("cpu"),
        )
        assert False, "Expected ValueError when final_grid is missing in stage2."
    except ValueError as exc:
        assert "final_grid" in str(exc)


def test_warp_backend_requires_warp_stats_key() -> None:
    class NoWarpStatsBackend(MockWarpBackend):
        def warp(self, image: torch.Tensor, params: torch.Tensor, residual_flow_lowres: torch.Tensor | None):
            out = super().warp(image, params, residual_flow_lowres)
            out.pop("warp_stats", None)
            return out

    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny"))
    loss_fn = CompositeLoss(config_for_stage("stage1_param_only"))
    stage = get_stage_toggles("stage1_param_only")
    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))

    try:
        _ = run_train_step(
            model=model,
            batch=_dummy_batch(),
            loss_fn=loss_fn,
            warp_backend=NoWarpStatsBackend(),
            stage=stage,
            optimizer=optimizer,
            scaler=None,
            amp_enabled=False,
            grad_clip_norm=1.0,
            device=torch.device("cpu"),
        )
        assert False, "Expected ValueError when warp_stats is missing."
    except ValueError as exc:
        assert "warp_stats" in str(exc)
