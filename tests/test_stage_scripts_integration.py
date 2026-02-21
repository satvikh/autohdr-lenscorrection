from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _train_common import run_training_from_configs
from src.losses.composite import CompositeLoss, config_for_stage
from src.models.hybrid_model import HybridLensCorrectionModel, HybridModelConfig
from src.train.checkpointing import load_checkpoint
from src.train.engine import EngineConfig, TrainerEngine
from src.train.optim import OptimConfig, SchedulerConfig, create_optimizer, create_scheduler
from src.train.stage_configs import get_stage_toggles
from src.train.warp_backends import MockWarpBackend


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_cfg(path: Path) -> None:
    _write_text(
        path,
        "\n".join(
            [
                "backbone_name: tiny",
                "pretrained_backbone: false",
                "use_coord_channels: true",
                "include_aspect: false",
                "param_hidden_dim: 64",
                "residual_hidden_dim: 32",
                "residual_max_disp: 4.0",
                "return_pred_image: false",
                "param_bounds:",
                "  k1: [-0.6, 0.6]",
                "  k2: [-0.3, 0.3]",
                "  k3: [-0.15, 0.15]",
                "  p1: [-0.03, 0.03]",
                "  p2: [-0.03, 0.03]",
                "  dcx: [-0.08, 0.08]",
                "  dcy: [-0.08, 0.08]",
                "  scale: [0.90, 1.20]",
                "  aspect: [0.97, 1.03]",
            ]
        ),
    )


def _loss_cfg(path: Path, *, stage: str) -> None:
    if stage == "stage1_param_only":
        flow_tv = 0.0
        flow_mag = 0.0
        flow_curv = 0.0
        jacobian = 0.0
    else:
        flow_tv = 0.02
        flow_mag = 0.01
        flow_curv = 0.005
        jacobian = 0.01

    _write_text(
        path,
        "\n".join(
            [
                f"stage: {stage}",
                "use_charbonnier: true",
                "charbonnier_eps: 0.001",
                "multiscale_scales: [1.0]",
                "pixel_weight: 0.10",
                "ssim_weight: 0.15",
                "edge_weight: 0.40",
                "grad_orient_weight: 0.18",
                f"flow_tv_weight: {flow_tv}",
                f"flow_mag_weight: {flow_mag}",
                f"flow_curv_weight: {flow_curv}",
                f"jacobian_weight: {jacobian}",
                "jacobian_margin: 0.0",
            ]
        ),
    )


def _train_cfg(
    path: Path,
    *,
    stage: str,
    run_name: str,
    checkpoint_dir: Path,
    epochs: int,
    max_steps: int,
    max_val_steps: int,
    proxy_enabled: bool = False,
    proxy_module_path: str = "src.metrics.proxy_score",
    best_metric_name: str = "total",
    best_metric_mode: str = "min",
) -> None:
    _write_text(
        path,
        "\n".join(
            [
                f"run_name: {run_name}",
                f"stage: {stage}",
                "warp_backend: mock",
                "synthetic_train_steps: 2",
                "synthetic_val_steps: 1",
                "synthetic_batch_size: 2",
                "synthetic_height: 48",
                "synthetic_width: 64",
                "device: cpu",
                f"epochs: {epochs}",
                f"max_steps_per_epoch: {max_steps}",
                f"max_val_steps: {max_val_steps}",
                "log_interval: 1",
                "amp_enabled: false",
                "grad_clip_norm: 1.0",
                f"checkpoint_dir: {checkpoint_dir.as_posix()}",
                f"proxy_enabled: {'true' if proxy_enabled else 'false'}",
                f"proxy_module_path: {proxy_module_path}",
                "proxy_function_name: compute_proxy_score",
                "proxy_config: {}",
                f"best_metric_name: {best_metric_name}",
                f"best_metric_mode: {best_metric_mode}",
                "optimizer:",
                "  lr: 0.0003",
                "  weight_decay: 0.0001",
                "  betas: [0.9, 0.999]",
                "scheduler:",
                "  name: none",
                "  min_lr: 0.000001",
            ]
        ),
    )


def test_training_entrypoint_synthetic_validate_only_and_proxy_fallback(tmp_path: Path) -> None:
    model_path = tmp_path / "model.yaml"
    loss_path = tmp_path / "loss.yaml"
    train_path = tmp_path / "train.yaml"
    ckpt_dir = tmp_path / "runs"

    _model_cfg(model_path)
    _loss_cfg(loss_path, stage="stage1_param_only")
    _train_cfg(
        train_path,
        stage="stage1_param_only",
        run_name="stage1_syn",
        checkpoint_dir=ckpt_dir,
        epochs=1,
        max_steps=1,
        max_val_steps=1,
        proxy_enabled=True,
        proxy_module_path="nonexistent.proxy.module",
    )

    train_metrics = run_training_from_configs(
        model_config_path=str(model_path),
        loss_config_path=str(loss_path),
        train_config_path=str(train_path),
        use_synthetic=True,
        loader_module=None,
        loader_fn="build_train_val_loaders",
        run_name_override="stage1_syn",
    )
    assert math.isfinite(float(train_metrics["train_total"]))
    assert math.isfinite(float(train_metrics["val_total"]))
    assert math.isnan(float(train_metrics["val_proxy_total"]))

    validate_metrics = run_training_from_configs(
        model_config_path=str(model_path),
        loss_config_path=str(loss_path),
        train_config_path=str(train_path),
        use_synthetic=True,
        loader_module=None,
        loader_fn="build_train_val_loaders",
        validate_only=True,
        run_name_override="stage1_validate_only",
    )
    assert math.isnan(float(validate_metrics["train_total"]))
    assert math.isfinite(float(validate_metrics["val_total"]))

    assert (ckpt_dir / "stage1_syn" / "best.pt").exists()
    assert (ckpt_dir / "stage1_syn" / "last.pt").exists()


def test_training_entrypoint_external_loader_resume_and_init(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "loader_ext.py"
    _write_text(
        module_path,
        "\n".join(
            [
                "import torch",
                "",
                "def build_train_val_loaders(stage: str):",
                "    _ = stage",
                "    train = [{",
                "        'input_image': torch.rand(2, 3, 48, 64, dtype=torch.float32),",
                "        'target_image': torch.rand(2, 3, 48, 64, dtype=torch.float32),",
                "    }]",
                "    val = [{",
                "        'input_image': torch.rand(2, 3, 48, 64, dtype=torch.float32),",
                "        'target_image': torch.rand(2, 3, 48, 64, dtype=torch.float32),",
                "    }]",
                "    return train, val",
            ]
        ),
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    ckpt_dir = tmp_path / "runs"
    model_path = tmp_path / "model.yaml"
    _model_cfg(model_path)

    s1_loss = tmp_path / "loss_s1.yaml"
    s1_train_epoch2 = tmp_path / "train_s1_epoch2.yaml"
    s1_train_epoch3 = tmp_path / "train_s1_epoch3.yaml"
    _loss_cfg(s1_loss, stage="stage1_param_only")
    _train_cfg(
        s1_train_epoch2,
        stage="stage1_param_only",
        run_name="stage1_external",
        checkpoint_dir=ckpt_dir,
        epochs=2,
        max_steps=1,
        max_val_steps=1,
    )
    _train_cfg(
        s1_train_epoch3,
        stage="stage1_param_only",
        run_name="stage1_external_resumed",
        checkpoint_dir=ckpt_dir,
        epochs=3,
        max_steps=1,
        max_val_steps=1,
    )

    _ = run_training_from_configs(
        model_config_path=str(model_path),
        loss_config_path=str(s1_loss),
        train_config_path=str(s1_train_epoch2),
        use_synthetic=False,
        loader_module="loader_ext",
        loader_fn="build_train_val_loaders",
        run_name_override="stage1_external",
    )

    resume_from = ckpt_dir / "stage1_external" / "last.pt"
    assert resume_from.exists()

    _ = run_training_from_configs(
        model_config_path=str(model_path),
        loss_config_path=str(s1_loss),
        train_config_path=str(s1_train_epoch3),
        use_synthetic=False,
        loader_module="loader_ext",
        loader_fn="build_train_val_loaders",
        resume_from=str(resume_from),
        run_name_override="stage1_external_resumed",
    )
    assert (ckpt_dir / "stage1_external_resumed" / "last.pt").exists()

    s2_loss = tmp_path / "loss_s2.yaml"
    s2_train = tmp_path / "train_s2.yaml"
    _loss_cfg(s2_loss, stage="stage2_hybrid")
    _train_cfg(
        s2_train,
        stage="stage2_hybrid",
        run_name="stage2_init",
        checkpoint_dir=ckpt_dir,
        epochs=1,
        max_steps=1,
        max_val_steps=1,
    )

    _ = run_training_from_configs(
        model_config_path=str(model_path),
        loss_config_path=str(s2_loss),
        train_config_path=str(s2_train),
        use_synthetic=False,
        loader_module="loader_ext",
        loader_fn="build_train_val_loaders",
        init_from=str(ckpt_dir / "stage1_external" / "best.pt"),
        run_name_override="stage2_init",
    )
    assert (ckpt_dir / "stage2_init" / "last.pt").exists()


def test_best_checkpoint_selection_can_use_proxy_metric(tmp_path: Path) -> None:
    class ScriptedValidationTrainer(TrainerEngine):
        def __init__(self, *, scripted_val: list[dict[str, float]], **kwargs):
            super().__init__(**kwargs)
            self.scripted_val = scripted_val

        def train_one_epoch(self, train_loader, epoch: int) -> dict[str, float]:
            _ = train_loader
            _ = epoch
            return {"total": 1.0}

        def validate(self, val_loader, epoch: int) -> dict[str, float]:
            _ = val_loader
            return self.scripted_val[epoch - 1]

    model = HybridLensCorrectionModel(config=HybridModelConfig(backbone_name="tiny"))
    loss_fn = CompositeLoss(config_for_stage("stage1_param_only"))
    stage = get_stage_toggles("stage1_param_only")
    optimizer = create_optimizer(model, OptimConfig(lr=1e-3, weight_decay=0.0))
    sched_bundle = create_scheduler(optimizer, SchedulerConfig(name="none"), total_steps=2, steps_per_epoch=1)

    trainer = ScriptedValidationTrainer(
        model=model,
        loss_fn=loss_fn,
        stage=stage,
        warp_backend=MockWarpBackend(),
        optimizer=optimizer,
        scheduler=sched_bundle.scheduler,
        scheduler_step_interval=sched_bundle.step_interval,
        config=EngineConfig(
            epochs=2,
            device="cpu",
            checkpoint_dir=str(tmp_path / "runs"),
            best_metric_name="proxy_total_score",
            best_metric_mode="max",
        ),
        scripted_val=[
            {"total": 0.20, "proxy_total_score": 0.40},
            {"total": 0.35, "proxy_total_score": 0.90},
        ],
    )

    out = trainer.fit(train_loader=[{}], val_loader=[{}], run_name="proxy_best")
    assert out["best_metric_name"] == "proxy_total_score"
    assert out["best_metric_mode"] == "max"
    assert abs(float(out["best_metric_value"]) - 0.90) < 1e-8

    meta = load_checkpoint(tmp_path / "runs" / "proxy_best" / "best.pt", model=model, map_location="cpu")
    assert int(meta["epoch"]) == 2
    assert abs(float(meta["best_metric"]) - 0.90) < 1e-8
