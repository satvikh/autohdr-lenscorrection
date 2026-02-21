from __future__ import annotations

import pytest

from src.train.config_loader import load_loss_config, load_model_config, load_train_config


def test_model_config_loader_parses_bounds() -> None:
    model_cfg, bounds = load_model_config("configs/model/debug_smoke.yaml")
    assert model_cfg.backbone_name == "tiny"
    assert model_cfg.use_coord_channels is True
    assert bounds.k1[0] < 0.0 < bounds.k1[1]


def test_loss_config_loader_parses_stage() -> None:
    loss_cfg = load_loss_config("configs/loss/stage2_hybrid.yaml")
    assert loss_cfg.stage == "stage2_hybrid"
    assert loss_cfg.flow_tv_weight > 0.0


def test_train_config_loader_parses_proxy_fields() -> None:
    engine_cfg, _, _, extra = load_train_config("configs/train/debug_smoke.yaml")
    assert engine_cfg.proxy_enabled is False
    assert engine_cfg.proxy_function_name == "compute_proxy_score"
    assert engine_cfg.fail_on_nonfinite_loss is True
    assert engine_cfg.param_saturation_warn_threshold > 0.0
    assert engine_cfg.residual_warn_abs_max_px > 0.0
    assert extra["warp_backend"] in {"person1", "mock"}


def test_train_config_loader_rejects_invalid_stage(tmp_path) -> None:
    p = tmp_path / "bad_train.yaml"
    p.write_text(
        "\n".join(
            [
                "run_name: bad",
                "stage: bogus_stage",
                "warp_backend: person1",
                "optimizer: {lr: 0.0001, weight_decay: 0.0, betas: [0.9, 0.999]}",
                "scheduler: {name: none}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train.stage"):
        _ = load_train_config(p)


def test_train_config_loader_rejects_invalid_warp_backend(tmp_path) -> None:
    p = tmp_path / "bad_warp.yaml"
    p.write_text(
        "\n".join(
            [
                "run_name: bad",
                "stage: stage1_param_only",
                "warp_backend: unknown_backend",
                "optimizer: {lr: 0.0001, weight_decay: 0.0, betas: [0.9, 0.999]}",
                "scheduler: {name: none}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="warp_backend"):
        _ = load_train_config(p)
