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
    assert engine_cfg.proxy_fullres_slice_enabled is False
    assert engine_cfg.proxy_fullres_max_images == 0
    assert engine_cfg.fail_on_nonfinite_loss is True
    assert engine_cfg.param_saturation_warn_threshold > 0.0
    assert engine_cfg.residual_warn_abs_max_px > 0.0
    assert extra["warp_backend"] in {"person1", "mock"}
    assert engine_cfg.best_metric_name == "total"
    assert engine_cfg.best_metric_mode == "min"
    assert engine_cfg.debug_instrumentation is False
    assert engine_cfg.debug_probe_enabled is False


def test_train_config_loader_rejects_invalid_best_metric_mode(tmp_path) -> None:
    p = tmp_path / "bad_best_metric.yaml"
    p.write_text(
        "\n".join(
            [
                "run_name: bad",
                "stage: stage1_param_only",
                "warp_backend: person1",
                "best_metric_name: proxy_total_score",
                "best_metric_mode: highest",
                "optimizer: {lr: 0.0001, weight_decay: 0.0, betas: [0.9, 0.999]}",
                "scheduler: {name: none}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="best_metric_mode"):
        _ = load_train_config(p)


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


def test_train_config_loader_rejects_invalid_debug_precision(tmp_path) -> None:
    p = tmp_path / "bad_debug_precision.yaml"
    p.write_text(
        "\n".join(
            [
                "run_name: bad",
                "stage: stage1_param_only",
                "warp_backend: person1",
                "debug_metric_precision: 0",
                "optimizer: {lr: 0.0001, weight_decay: 0.0, betas: [0.9, 0.999]}",
                "scheduler: {name: none}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="debug_metric_precision"):
        _ = load_train_config(p)


def test_train_config_loader_rejects_negative_proxy_fullres_max_images(tmp_path) -> None:
    p = tmp_path / "bad_proxy_fullres_max_images.yaml"
    p.write_text(
        "\n".join(
            [
                "run_name: bad",
                "stage: stage1_param_only",
                "warp_backend: person1",
                "proxy_fullres_max_images: -1",
                "optimizer: {lr: 0.0001, weight_decay: 0.0, betas: [0.9, 0.999]}",
                "scheduler: {name: none}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="proxy_fullres_max_images"):
        _ = load_train_config(p)


def test_train_config_loader_applies_default_proxy_config_when_enabled(tmp_path) -> None:
    p = tmp_path / "proxy_default.yaml"
    p.write_text(
        "\n".join(
            [
                "run_name: proxy_default",
                "stage: stage1_param_only",
                "warp_backend: person1",
                "proxy_enabled: true",
                "proxy_module_path: src.metrics.proxy_score",
                "proxy_function_name: compute_proxy_score",
                "proxy_config: {}",
                "optimizer: {lr: 0.0001, weight_decay: 0.0, betas: [0.9, 0.999]}",
                "scheduler: {name: none}",
            ]
        ),
        encoding="utf-8",
    )
    engine_cfg, _, _, _ = load_train_config(p)
    assert engine_cfg.proxy_enabled is True
    assert isinstance(engine_cfg.proxy_config, dict)
    assert engine_cfg.proxy_config["weights"]["edge"] == 0.40
    assert engine_cfg.proxy_config["aggregation"]["fail_policy"] == "score_zero"
    assert engine_cfg.proxy_config["hardfail"]["penalty_mode"] == "score_zero"


def test_train_config_loader_merges_proxy_config_overrides(tmp_path) -> None:
    p = tmp_path / "proxy_override.yaml"
    p.write_text(
        "\n".join(
            [
                "run_name: proxy_override",
                "stage: stage1_param_only",
                "warp_backend: person1",
                "proxy_enabled: true",
                "proxy_module_path: src.metrics.proxy_score",
                "proxy_function_name: compute_proxy_score",
                "proxy_config:",
                "  hardfail:",
                "    edge_min: 0.07",
                "    penalty_mode: clamp",
                "    penalty_value: 0.04",
                "optimizer: {lr: 0.0001, weight_decay: 0.0, betas: [0.9, 0.999]}",
                "scheduler: {name: none}",
            ]
        ),
        encoding="utf-8",
    )
    engine_cfg, _, _, _ = load_train_config(p)
    assert isinstance(engine_cfg.proxy_config, dict)
    assert abs(float(engine_cfg.proxy_config["hardfail"]["edge_min"]) - 0.07) < 1e-9
    assert engine_cfg.proxy_config["hardfail"]["penalty_mode"] == "clamp"
    assert abs(float(engine_cfg.proxy_config["hardfail"]["penalty_value"]) - 0.04) < 1e-9
    # Unspecified defaults remain present.
    assert engine_cfg.proxy_config["weights"]["line"] == 0.22
