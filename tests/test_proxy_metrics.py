"""Tests for proxy metric entrypoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.metrics.proxy_edge import compute_edge_score
from src.metrics.proxy_score import aggregate_scores, compute_proxy_score
from src.metrics.proxy_ssim_mae import compute_mae, compute_ssim


def _base_config() -> dict:
    return {
        "weights": {"edge": 0.40, "line": 0.22, "grad": 0.18, "ssim": 0.15, "mae": 0.05},
        "hardfail": {"penalty_mode": "clamp", "penalty_value": 0.05, "mae_max": 0.9},
        "aggregation": {"fail_policy": "exclude"},
    }


def test_proxy_score_schema_and_range() -> None:
    gt = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    pred = gt.copy()
    score = compute_proxy_score(pred, gt, _base_config())

    assert {"total_score", "sub_scores", "flags", "total", "subscores"}.issubset(set(score.keys()))
    assert set(score["sub_scores"].keys()) == {"edge", "line", "grad", "ssim", "mae"}
    assert set(score["subscores"].keys()) == {"edge", "line", "grad", "ssim", "mae"}
    assert set(score["flags"].keys()) == {"hard_fail", "hardfail", "reasons"}
    assert 0.0 <= score["total_score"] <= 1.0
    assert score["total_score"] == score["total"]
    assert score["flags"]["hard_fail"] is False
    assert score["flags"]["hardfail"] is False


def test_proxy_score_degrades_for_noisy_prediction() -> None:
    rng = np.random.default_rng(42)
    gt = np.ones((48, 48, 3), dtype=np.float32) * 0.5
    clean = compute_proxy_score(gt.copy(), gt, _base_config())["total_score"]
    noisy_pred = np.clip(gt + rng.normal(0.0, 0.3, size=gt.shape), 0.0, 1.0).astype(np.float32)
    noisy = compute_proxy_score(noisy_pred, gt, _base_config())["total_score"]

    assert noisy < clean


def test_proxy_hardfail_on_invalid_values() -> None:
    gt = np.ones((16, 16, 3), dtype=np.float32)
    pred = gt.copy()
    pred[0, 0, 0] = np.nan

    score = compute_proxy_score(pred, gt, _base_config())
    assert score["flags"]["hard_fail"] is True
    assert score["flags"]["hardfail"] is True
    assert "non_finite_values" in score["flags"]["reasons"]


def test_aggregate_scores_summary() -> None:
    gt = np.ones((8, 8, 3), dtype=np.float32)
    rows = [compute_proxy_score(gt, gt, _base_config()) for _ in range(3)]
    summary = aggregate_scores(rows, _base_config())

    assert summary["count"] == 3
    assert 0.0 <= summary["mean_total"] <= 1.0
    assert set(summary["mean_subscores"].keys()) == {"edge", "line", "grad", "ssim", "mae"}


def test_mae_ssim_identical_images() -> None:
    gt = np.ones((32, 32, 3), dtype=np.float32) * 0.4
    pred = gt.copy()

    mae = compute_mae(pred, gt)
    ssim = compute_ssim(pred, gt)

    assert isinstance(mae, float)
    assert isinstance(ssim, float)
    assert mae <= 1e-8
    assert ssim >= 0.99


def test_mae_ssim_inverted_images() -> None:
    gt = np.linspace(0.0, 1.0, num=32 * 32 * 3, dtype=np.float32).reshape((32, 32, 3))
    pred = 1.0 - gt

    mae = compute_mae(pred, gt)
    ssim = compute_ssim(pred, gt)
    ssim_identical = compute_ssim(gt, gt)

    assert mae > 0.4
    assert ssim < ssim_identical


def test_mae_ssim_deterministic() -> None:
    rng = np.random.default_rng(123)
    gt = rng.random((24, 24, 3), dtype=np.float32)
    pred = rng.random((24, 24, 3), dtype=np.float32)

    mae_1 = compute_mae(pred, gt)
    mae_2 = compute_mae(pred, gt)
    ssim_1 = compute_ssim(pred, gt)
    ssim_2 = compute_ssim(pred, gt)

    assert mae_1 == mae_2
    assert ssim_1 == ssim_2


def test_edge_score_identical_images_near_one() -> None:
    rng = np.random.default_rng(7)
    img = rng.random((40, 40, 3), dtype=np.float32)
    score = compute_edge_score(img, img.copy(), {})
    assert score >= 0.99


def test_edge_score_blank_vs_textured_is_lower() -> None:
    blank = np.zeros((48, 48, 3), dtype=np.float32)
    textured = np.zeros((48, 48, 3), dtype=np.float32)
    textured[:, ::2, :] = 1.0

    score_same = compute_edge_score(blank, blank.copy(), {})
    score_mismatch = compute_edge_score(blank, textured, {})

    assert score_same >= 0.99
    assert score_mismatch < score_same


def test_hardfail_fail_policy_behavior() -> None:
    gt = np.ones((16, 16, 3), dtype=np.float32)
    pred = gt.copy()
    pred[0, 0, 0] = np.nan

    cases = [
        ("exclude", float("nan")),
        ("score_zero", 0.0),
        ("score_neg_inf", -1e9),
    ]
    for policy, expected_total in cases:
        cfg = _base_config()
        cfg["aggregation"] = {"fail_policy": policy}
        out = compute_proxy_score(pred, gt, cfg)

        assert {"total_score", "sub_scores", "flags", "total", "subscores"}.issubset(set(out.keys()))
        assert out["flags"]["hard_fail"] is True
        assert out["flags"]["hardfail"] is True
        assert set(out["subscores"].keys()) == {"edge", "line", "grad", "ssim", "mae"}
        assert set(out["sub_scores"].keys()) == {"edge", "line", "grad", "ssim", "mae"}
        assert all(np.isnan(v) for v in out["subscores"].values())
        assert all(np.isnan(v) for v in out["sub_scores"].values())
        if np.isnan(expected_total):
            assert np.isnan(out["total_score"])
            assert np.isnan(out["total"])
        else:
            assert out["total_score"] == expected_total
            assert out["total"] == expected_total


def test_weighting_with_disabled_metric_is_renormalized() -> None:
    gt = np.zeros((32, 32, 3), dtype=np.float32)
    pred = np.ones((32, 32, 3), dtype=np.float32) * 0.5

    cfg_with_edge = _base_config()
    cfg_with_edge["metrics"] = {"edge": True, "mae": True, "line": False, "grad": False, "ssim": False}
    cfg_with_edge["weights"] = {"edge": 0.9, "mae": 0.1}
    score_with_edge = compute_proxy_score(pred, gt, cfg_with_edge)

    cfg_mae_only = _base_config()
    cfg_mae_only["metrics"] = {"edge": False, "mae": True, "line": False, "grad": False, "ssim": False}
    cfg_mae_only["weights"] = {"edge": 0.9, "mae": 0.1}
    score_mae_only = compute_proxy_score(pred, gt, cfg_mae_only)

    # mae=0.5 -> mae_score=0.5 when it is the only enabled metric.
    assert abs(score_mae_only["total_score"] - 0.5) < 1e-6
    assert score_with_edge["total_score"] > score_mae_only["total_score"]


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    img.save(path)


def test_validate_proxy_smoke(tmp_path: Path) -> None:
    from scripts.validate_proxy import main as validate_main

    pred_dir = tmp_path / "pred"
    gt_root = tmp_path / "gt"
    out_dir = tmp_path / "reports"
    split_csv = tmp_path / "split.csv"
    config_path = tmp_path / "config.json"

    img1 = np.ones((16, 16, 3), dtype=np.float32) * 0.25
    img2 = np.ones((16, 16, 3), dtype=np.float32) * 0.75

    _write_png(pred_dir / "img_1.png", img1)
    _write_png(pred_dir / "img_2.png", img2)
    _write_png(gt_root / "img_1.png", img1)
    _write_png(gt_root / "img_2.png", img2)

    split_csv.write_text(
        "image_id,rel_target_path\n"
        "img_1,img_1.png\n"
        "img_2,img_2.png\n",
        encoding="utf-8",
    )

    config_path.write_text(
        json.dumps(
            {
                "image": {"allowed_ext": [".png"], "require_same_size": True},
                "aggregation": {"fail_policy": "exclude", "allowed_fail_rate": 0.0},
            }
        ),
        encoding="utf-8",
    )

    exit_code = validate_main(
        [
            "--pred_dir",
            str(pred_dir),
            "--split_csv",
            str(split_csv),
            "--gt_root",
            str(gt_root),
            "--config",
            str(config_path),
            "--out_dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    per_image_csv = out_dir / "per_image_scores.csv"
    summary_json = out_dir / "summary.json"
    assert per_image_csv.exists()
    assert summary_json.exists()

    with per_image_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["count"] == 2


def test_validate_proxy_resolves_target_path_relative_to_cwd(tmp_path: Path, monkeypatch) -> None:
    from scripts.validate_proxy import main as validate_main

    pred_dir = tmp_path / "pred"
    out_dir = tmp_path / "reports"
    split_dir = tmp_path / "splits"
    split_csv = split_dir / "split.csv"
    config_path = tmp_path / "config.json"
    gt_dir = tmp_path / "gt"

    img = np.ones((10, 12, 3), dtype=np.float32) * 0.4
    _write_png(pred_dir / "img_1.png", img)
    _write_png(gt_dir / "img_1.png", img)

    split_dir.mkdir(parents=True, exist_ok=True)
    split_csv.write_text(
        "image_id,target_path\n"
        "img_1,gt/img_1.png\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "image": {"allowed_ext": [".png"]},
                "aggregation": {"fail_policy": "exclude", "allowed_fail_rate": 0.0},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    exit_code = validate_main(
        [
            "--pred_dir",
            str(pred_dir),
            "--split_csv",
            str(split_csv),
            "--config",
            str(config_path),
            "--out_dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
