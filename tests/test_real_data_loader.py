from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from src.data.real_loader import build_split_csvs, build_train_val_loaders


def _write_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=color).save(path)


def _make_mock_real_dataset(root: Path) -> Path:
    pairs: list[dict] = []
    # 3 photoshoots x 3 frames each.
    for shoot_idx in range(3):
        shoot = f"shoot_{shoot_idx}"
        for frame_idx in range(3):
            pair_id = f"{shoot}_g{frame_idx}"
            orig_name = f"{pair_id}_original.jpg"
            gen_name = f"{pair_id}_generated.jpg"
            _write_rgb(root / orig_name, size=(40, 28), color=(10 + shoot_idx, 20 + frame_idx, 30))
            _write_rgb(root / gen_name, size=(40, 28), color=(40 + shoot_idx, 50 + frame_idx, 60))
            pairs.append(
                {
                    "pair_id": pair_id,
                    "original": orig_name,
                    "generated": gen_name,
                    "photoshoot_uuid": shoot,
                }
            )

    payload = {"total_pairs": len(pairs), "pairs": pairs}
    metadata_path = root / "training_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path


def _read_csv_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [str(r["image_id"]) for r in reader]


def test_build_split_csvs_from_metadata(tmp_path: Path) -> None:
    root = tmp_path / "train_root"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = _make_mock_real_dataset(root)
    split_dir = tmp_path / "splits"

    report = build_split_csvs(
        root_dir=root,
        split_dir=split_dir,
        metadata_path=metadata_path,
        val_frac=0.2,
        seed=123,
        max_pairs=None,
    )

    train_csv = split_dir / "train_split.csv"
    val_csv = split_dir / "val_split.csv"
    assert train_csv.exists()
    assert val_csv.exists()
    assert int(report["total_pairs"]) == 9
    assert int(report["train_pairs"]) + int(report["val_pairs"]) == 9

    train_ids = set(_read_csv_ids(train_csv))
    val_ids = set(_read_csv_ids(val_csv))
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) > 0
    assert len(val_ids) > 0


def test_build_train_val_loaders_respects_env_contract(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "train_root"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = _make_mock_real_dataset(root)
    split_dir = tmp_path / "splits"

    monkeypatch.setenv("AUTOHDR_DATA_ROOT", str(root))
    monkeypatch.setenv("AUTOHDR_TRAIN_METADATA", str(metadata_path))
    monkeypatch.setenv("AUTOHDR_SPLIT_DIR", str(split_dir))
    monkeypatch.setenv("AUTOHDR_REBUILD_SPLITS", "1")
    monkeypatch.setenv("AUTOHDR_BATCH_SIZE", "2")
    monkeypatch.setenv("AUTOHDR_NUM_WORKERS", "0")
    monkeypatch.setenv("AUTOHDR_RESIZE_HW", "24,36")
    monkeypatch.setenv("AUTOHDR_TRAIN_HFLIP", "0.0")
    monkeypatch.setenv("AUTOHDR_MAX_PAIRS", "8")

    train_loader, val_loader = build_train_val_loaders(stage="stage1_param_only")
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    for batch in (train_batch, val_batch):
        assert {"input_image", "target_image", "image_id", "orig_size"}.issubset(set(batch.keys()))
        assert batch["input_image"].ndim == 4
        assert batch["target_image"].ndim == 4
        assert tuple(batch["input_image"].shape[-2:]) == (24, 36)
        assert tuple(batch["target_image"].shape[-2:]) == (24, 36)
