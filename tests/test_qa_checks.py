"""Tests for QA filename and image integrity checks."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image

from src.qa.filename_check import check_filenames
from src.qa.image_integrity_check import check_images


def _write_rgb(path: Path, size=(16, 16), color=(128, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size=size, color=color)
    img.save(path)


def _filename_cfg(allowed_ext, allow_extra_files=False) -> dict:
    return {"image": {"allowed_ext": allowed_ext}, "qa": {"allow_extra_files": allow_extra_files}}


def test_check_filenames_reports_missing_file(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    _write_rgb(pred / "img_a.png")

    result = check_filenames(pred, ["img_a", "img_b"], _filename_cfg([".png"]))

    assert result["ok"] is False
    assert result["missing"] == ["img_b"]
    assert result["extra"] == []
    assert result["bad_names"] == []


def test_check_filenames_reports_extra_when_not_allowed(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    _write_rgb(pred / "img_a.png")
    _write_rgb(pred / "img_extra.png")

    result = check_filenames(pred, ["img_a"], _filename_cfg([".png"], allow_extra_files=False))

    assert result["ok"] is False
    assert "img_extra.png" in result["extra"]


def test_check_filenames_reports_bad_extension_name(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    _write_rgb(pred / "img_a.jpg")

    result = check_filenames(pred, ["img_a"], _filename_cfg([".png"]))

    assert result["ok"] is False
    assert "img_a" in result["missing"]
    assert "img_a.jpg" in result["bad_names"]


def test_check_filenames_detects_duplicate_allowed_extensions(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    _write_rgb(pred / "img_a.jpg")
    _write_rgb(pred / "img_a.png")
    result = check_filenames(pred, ["img_a"], _filename_cfg([".jpg", ".png"]))
    assert result["ok"] is False
    assert "img_a.jpg" in result["bad_names"]
    assert "img_a.png" in result["bad_names"]


def test_check_filenames_ignores_non_image_aux_files_by_default(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    _write_rgb(pred / "img_a.png")
    (pred / "run_metadata.json").write_text("{}", encoding="utf-8")

    result = check_filenames(pred, ["img_a"], _filename_cfg([".png"]))
    assert result["ok"] is True
    assert result["bad_names"] == []


def test_check_images_detects_size_mismatch(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"

    _write_rgb(pred / "img_1.jpg", size=(20, 20))
    _write_rgb(gt / "img_1.png", size=(16, 16))

    result = check_images(
        pred,
        ["img_1"],
        gt_root=gt,
        config={"allowed_ext": [".jpg"], "image": {"require_same_size": True}},
    )

    assert result["ok"] is False
    reasons = [r["reason"] for r in result["bad_files"] if r["image_id"] == "img_1"]
    assert any(reason.startswith("size_mismatch:") for reason in reasons)


def test_check_images_ok_for_matching_images(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"

    _write_rgb(pred / "img_1.jpg", size=(12, 12))
    _write_rgb(gt / "img_1.png", size=(12, 12))

    result = check_images(pred, ["img_1"], gt_root=gt, config={"allowed_ext": [".jpg"]})

    assert result["ok"] is True
    assert result["bad_files"] == []


def test_check_images_flags_corrupt_image(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir(parents=True, exist_ok=True)
    corrupt = pred / "img_bad.jpg"
    corrupt.write_bytes(b"not_an_image")

    result = check_images(pred, ["img_bad"], config={"allowed_ext": [".jpg"]})

    assert result["ok"] is False
    reasons = [r["reason"] for r in result["bad_files"] if r["image_id"] == "img_bad"]
    assert any(reason.startswith("pred_load_error:") for reason in reasons)


def test_build_submission_zip_strict_blocks_on_qa_fail(tmp_path: Path) -> None:
    from scripts.build_submission_zip import main as build_zip_main

    pred = tmp_path / "pred"
    pred.mkdir(parents=True, exist_ok=True)
    (pred / "img_1.png").write_bytes(b"corrupt_bytes")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("img_1\n", encoding="utf-8")
    out_zip = tmp_path / "submission.zip"

    exit_code = build_zip_main(
        [
            "--pred_dir",
            str(pred),
            "--ids_file",
            str(ids_file),
            "--out_zip",
            str(out_zip),
        ]
    )

    assert exit_code == 2
    assert not out_zip.exists()


def test_build_submission_zip_force_zip_creates_archive_on_qa_fail(tmp_path: Path) -> None:
    from scripts.build_submission_zip import main as build_zip_main

    pred = tmp_path / "pred"
    pred.mkdir(parents=True, exist_ok=True)
    (pred / "img_1.png").write_bytes(b"corrupt_bytes")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("img_1\n", encoding="utf-8")
    out_zip = tmp_path / "submission.zip"

    exit_code = build_zip_main(
        [
            "--pred_dir",
            str(pred),
            "--ids_file",
            str(ids_file),
            "--out_zip",
            str(out_zip),
            "--force_zip",
        ]
    )

    assert exit_code == 0
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip, "r") as zf:
        assert "img_1.png" in zf.namelist()

    qa_report = out_zip.with_name("submission_qa.json")
    assert qa_report.exists()
    report_data = json.loads(qa_report.read_text(encoding="utf-8"))
    assert report_data["ok"] is False
    assert "manifest" in report_data
    manifest = report_data["manifest"]
    assert "checkpoint_id" in manifest
    assert "config_path" in manifest
    assert "config_hash" in manifest
    assert "timestamp_utc" in manifest
    assert "git_commit" in manifest


def test_build_submission_zip_rejects_unsafe_zip_subdir(tmp_path: Path) -> None:
    from scripts.build_submission_zip import main as build_zip_main

    pred = tmp_path / "pred"
    pred.mkdir(parents=True, exist_ok=True)
    _write_rgb(pred / "img_1.png")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("img_1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "image": {"allowed_ext": [".png"]},
                "submission": {"zip_subdir": "../bad"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = build_zip_main(
        [
            "--pred_dir",
            str(pred),
            "--ids_file",
            str(ids_file),
            "--out_zip",
            str(tmp_path / "bad.zip"),
            "--config",
            str(config),
        ]
    )

    assert exit_code == 1


def test_build_submission_zip_manifest_falls_back_to_infer_metadata(tmp_path: Path) -> None:
    from scripts.build_submission_zip import main as build_zip_main

    pred = tmp_path / "pred"
    pred.mkdir(parents=True, exist_ok=True)
    _write_rgb(pred / "img_1.png")
    (pred / "run_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt_from_infer",
                "checkpoint_path": "outputs/runs/demo/best.pt",
                "model_source": "checkpoint",
                "config_hash": "abc123",
            }
        ),
        encoding="utf-8",
    )

    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("img_1\n", encoding="utf-8")
    out_zip = tmp_path / "submission.zip"

    exit_code = build_zip_main(
        [
            "--pred_dir",
            str(pred),
            "--ids_file",
            str(ids_file),
            "--out_zip",
            str(out_zip),
        ]
    )
    assert exit_code == 0

    qa_report = out_zip.with_name("submission_qa.json")
    report_data = json.loads(qa_report.read_text(encoding="utf-8"))
    manifest = report_data["manifest"]
    assert manifest["checkpoint_id"] == "ckpt_from_infer"
    assert manifest["checkpoint_path"] == "outputs/runs/demo/best.pt"
    assert manifest["model_source"] == "checkpoint"
    assert manifest["config_hash"] == "abc123"
