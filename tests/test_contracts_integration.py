from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import PairedLensDataset
from src.geometry.coords import make_identity_grid
from src.geometry.jacobian import jacobian_stats
from src.geometry.parametric_warp import build_parametric_grid
from src.geometry.residual_fusion import fuse_grids, upsample_residual_flow
from src.geometry.warp_ops import warp_image
from src.inference.safety import evaluate_safety
from src.metrics.proxy_score import compute_proxy_score
from src.models.hybrid_model import HybridLensCorrectionModel
from src.train.config_loader import load_loss_config, load_model_config, load_train_config


def _write_rgb(path: Path, hw: tuple[int, int], value: int) -> None:
    h, w = hw
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def test_dataset_contract_sample_keys_and_shapes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    target_path = tmp_path / "target.png"
    _write_rgb(input_path, (32, 48), 80)
    _write_rgb(target_path, (32, 48), 120)

    csv_path = tmp_path / "pairs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "input_path", "target_path"])
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "sample_001",
                "input_path": str(input_path),
                "target_path": str(target_path),
            }
        )

    ds = PairedLensDataset(csv_path)
    sample = ds[0]

    assert {"input_image", "target_image", "image_id", "orig_size"}.issubset(set(sample.keys()))
    assert tuple(sample["input_image"].shape) == (3, 32, 48)
    assert tuple(sample["target_image"].shape) == (3, 32, 48)
    assert sample["image_id"] == "sample_001"
    assert sample["orig_size"] == (32, 48)


def test_model_contract_output_keys_and_shapes() -> None:
    model_cfg, bounds = load_model_config("configs/model/debug_smoke.yaml")
    model = HybridLensCorrectionModel(config=model_cfg, param_bounds=bounds)
    x = torch.rand(2, 3, 64, 80, dtype=torch.float32)
    out = model(x)

    assert "params" in out
    assert "residual_flow" in out
    assert "residual_flow_lowres" in out
    assert "residual_flow_fullres" in out
    assert tuple(out["params"].shape) == (2, 8)
    assert out["residual_flow"].ndim == 4
    assert tuple(out["residual_flow_fullres"].shape[-2:]) == (64, 80)


def test_geometry_api_signatures_and_basic_callability() -> None:
    assert list(inspect.signature(build_parametric_grid).parameters.keys()) == [
        "params",
        "height",
        "width",
        "align_corners",
        "device",
        "dtype",
    ]
    assert list(inspect.signature(upsample_residual_flow).parameters.keys()) == [
        "flow_lr",
        "target_h",
        "target_w",
        "align_corners",
    ]
    assert list(inspect.signature(fuse_grids).parameters.keys()) == ["param_grid", "residual_flow"]
    assert list(inspect.signature(warp_image).parameters.keys()) == [
        "image",
        "grid",
        "mode",
        "padding_mode",
        "align_corners",
    ]
    assert list(inspect.signature(jacobian_stats).parameters.keys()) == ["grid"]

    params = torch.zeros((1, 8), dtype=torch.float32)
    params[:, 7] = 1.0
    grid = build_parametric_grid(params, 16, 20, True, torch.device("cpu"), torch.float32)
    assert tuple(grid.shape) == (1, 16, 20, 2)


def test_proxy_contract_keys_include_contract_and_legacy_aliases() -> None:
    gt = np.ones((24, 24, 3), dtype=np.float32) * 0.5
    score = compute_proxy_score(gt.copy(), gt, {})

    assert {"total_score", "sub_scores", "flags"}.issubset(set(score.keys()))
    assert {"total", "subscores"}.issubset(set(score.keys()))
    assert {"hard_fail", "reasons"}.issubset(set(score["flags"].keys()))
    assert {"hardfail"}.issubset(set(score["flags"].keys()))


def test_safety_contract_metric_aliases_present() -> None:
    grid = make_identity_grid(batch=1, height=12, width=12, dtype=torch.float32)
    residual = torch.zeros((1, 12, 12, 2), dtype=torch.float32)
    residual[..., 0] = 0.05
    residual[..., 1] = 0.10
    report = evaluate_safety(grid, residual_flow_norm_bhwc=residual)

    assert {"safe", "reasons", "metrics"}.issubset(set(report.keys()))
    metrics = report["metrics"]
    assert "oob_ratio" in metrics
    assert "border_invalid_ratio" in metrics
    assert "jacobian_negative_det_pct" in metrics
    assert "residual_magnitude" in metrics


def test_config_stage_alignment_for_stage_runs() -> None:
    stage_pairs = [
        ("configs/loss/stage1_param_only.yaml", "configs/train/stage1_param_only.yaml"),
        ("configs/loss/stage2_hybrid.yaml", "configs/train/stage2_hybrid.yaml"),
        ("configs/loss/stage3_finetune.yaml", "configs/train/stage3_finetune.yaml"),
    ]
    for loss_path, train_path in stage_pairs:
        loss_cfg = load_loss_config(loss_path)
        _, _, _, extra = load_train_config(train_path)
        assert loss_cfg.stage == extra["stage"]
