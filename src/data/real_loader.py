from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PairedLensDataset
from src.data.transforms import PairedTransformConfig, PairedTransforms


_PAIR_ID_PATTERN = re.compile(r"^(?P<base>.+)_g\d+$")


@dataclass(frozen=True)
class RealDataLoaderConfig:
    root_dir: Path
    split_dir: Path
    metadata_path: Path | None
    train_csv: Path
    val_csv: Path
    val_frac: float
    seed: int
    max_pairs: int | None
    batch_size: int
    num_workers: int
    resize_hw: tuple[int, int] | None
    train_hflip_prob: float
    pin_memory: bool
    rebuild_splits: bool


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return None
    out = int(raw)
    return out if out > 0 else None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _env_hw(name: str, default: tuple[int, int] | None) -> tuple[int, int] | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = str(raw).strip()
    if text == "":
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be 'H,W' or empty, got: {text!r}")
    h = int(parts[0])
    w = int(parts[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must have positive dimensions, got: {(h, w)}")
    return (h, w)


def resolve_real_data_config() -> RealDataLoaderConfig:
    root_dir = Path(os.getenv("AUTOHDR_DATA_ROOT", "automatic-lens-correction/lens-correction-train-cleaned"))
    split_dir = Path(os.getenv("AUTOHDR_SPLIT_DIR", "data/splits/real_pairs"))
    metadata_path_raw = os.getenv("AUTOHDR_TRAIN_METADATA", "")
    metadata_path: Path | None
    if metadata_path_raw.strip():
        metadata_path = Path(metadata_path_raw)
    else:
        default_meta = root_dir / "training_metadata.json"
        metadata_path = default_meta if default_meta.exists() else None

    val_frac = _env_float("AUTOHDR_VAL_FRAC", 0.10)
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"AUTOHDR_VAL_FRAC must be in (0,1), got {val_frac}")

    split_dir.mkdir(parents=True, exist_ok=True)
    return RealDataLoaderConfig(
        root_dir=root_dir,
        split_dir=split_dir,
        metadata_path=metadata_path,
        train_csv=split_dir / "train_split.csv",
        val_csv=split_dir / "val_split.csv",
        val_frac=val_frac,
        seed=_env_int("AUTOHDR_SPLIT_SEED", 123),
        max_pairs=_env_optional_int("AUTOHDR_MAX_PAIRS"),
        batch_size=_env_int("AUTOHDR_BATCH_SIZE", 2),
        num_workers=_env_int("AUTOHDR_NUM_WORKERS", 0),
        resize_hw=_env_hw("AUTOHDR_RESIZE_HW", (512, 768)),
        train_hflip_prob=_env_float("AUTOHDR_TRAIN_HFLIP", 0.5),
        pin_memory=_env_bool("AUTOHDR_PIN_MEMORY", False),
        rebuild_splits=_env_bool("AUTOHDR_REBUILD_SPLITS", False),
    )


def _group_from_pair_id(pair_id: str) -> str:
    m = _PAIR_ID_PATTERN.match(pair_id)
    if m is None:
        return pair_id
    return m.group("base")


def _scan_pairs_from_filenames(root_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for orig_path in sorted(root_dir.glob("*_original.jpg")):
        pair_id = orig_path.name[: -len("_original.jpg")]
        gen_path = root_dir / f"{pair_id}_generated.jpg"
        if not gen_path.exists():
            continue
        rows.append(
            {
                "image_id": pair_id,
                "input_path": str(orig_path),
                "target_path": str(gen_path),
                "group_id": _group_from_pair_id(pair_id),
            }
        )
    return rows


def _rows_from_metadata(root_dir: Path, metadata_path: Path) -> list[dict[str, str]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("training_metadata.json must contain a list field named 'pairs'")

    rows: list[dict[str, str]] = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        pair_id = str(p.get("pair_id", "")).strip()
        original = str(p.get("original", "")).strip()
        generated = str(p.get("generated", "")).strip()
        if not pair_id or not original or not generated:
            continue
        input_path = root_dir / original
        target_path = root_dir / generated
        if not input_path.exists() or not target_path.exists():
            continue
        group_id = str(p.get("photoshoot_uuid", "")).strip() or _group_from_pair_id(pair_id)
        rows.append(
            {
                "image_id": pair_id,
                "input_path": str(input_path),
                "target_path": str(target_path),
                "group_id": group_id,
            }
        )

    return rows


def _load_pair_rows(root_dir: Path, metadata_path: Path | None) -> list[dict[str, str]]:
    rows = _rows_from_metadata(root_dir, metadata_path) if metadata_path is not None else []
    if not rows:
        rows = _scan_pairs_from_filenames(root_dir)
    if not rows:
        raise ValueError(f"No valid training pairs found under: {root_dir}")

    dedup: dict[str, dict[str, str]] = {}
    for row in rows:
        dedup[row["image_id"]] = row
    return [dedup[k] for k in sorted(dedup)]


def _split_by_group(rows: list[dict[str, str]], val_frac: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    group_to_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group_to_rows.setdefault(row["group_id"], []).append(row)

    group_ids = sorted(group_to_rows)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    target_val = max(1, int(round(len(rows) * val_frac))) if len(rows) > 1 else 0
    val_rows: list[dict[str, str]] = []
    train_rows: list[dict[str, str]] = []

    for group_id in group_ids:
        bucket = group_to_rows[group_id]
        if len(val_rows) < target_val:
            val_rows.extend(bucket)
        else:
            train_rows.extend(bucket)

    if not train_rows and val_rows:
        train_rows.append(val_rows.pop())
    if not val_rows and train_rows:
        val_rows.append(train_rows.pop())

    return sorted(train_rows, key=lambda r: r["image_id"]), sorted(val_rows, key=lambda r: r["image_id"])


def build_split_csvs(
    *,
    root_dir: str | Path,
    split_dir: str | Path,
    metadata_path: str | Path | None = None,
    val_frac: float = 0.10,
    seed: int = 123,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Data root must be a directory: {root}")

    meta = Path(metadata_path) if metadata_path is not None else None
    if meta is not None and not meta.exists():
        raise FileNotFoundError(f"Metadata path not found: {meta}")

    rows = _load_pair_rows(root, meta)
    if max_pairs is not None and max_pairs > 0:
        rows = rows[:max_pairs]

    train_rows, val_rows = _split_by_group(rows, val_frac=val_frac, seed=seed)

    out_dir = Path(split_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train_split.csv"
    val_csv = out_dir / "val_split.csv"
    report_json = out_dir / "split_report.json"

    for path, payload in ((train_csv, train_rows), (val_csv, val_rows)):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "input_path", "target_path"])
            writer.writeheader()
            for row in payload:
                writer.writerow(
                    {
                        "image_id": row["image_id"],
                        "input_path": row["input_path"],
                        "target_path": row["target_path"],
                    }
                )

    report = {
        "total_pairs": len(rows),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "max_pairs": max_pairs,
        "root_dir": str(root),
        "metadata_path": str(meta) if meta is not None else None,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _ensure_split_csvs(cfg: RealDataLoaderConfig) -> None:
    if cfg.rebuild_splits or not cfg.train_csv.exists() or not cfg.val_csv.exists():
        build_split_csvs(
            root_dir=cfg.root_dir,
            split_dir=cfg.split_dir,
            metadata_path=cfg.metadata_path,
            val_frac=cfg.val_frac,
            seed=cfg.seed,
            max_pairs=cfg.max_pairs,
        )


def _build_train_transforms(cfg: RealDataLoaderConfig) -> PairedTransforms | None:
    if cfg.resize_hw is None and cfg.train_hflip_prob <= 0.0:
        return None
    tcfg = PairedTransformConfig(
        resize_hw=cfg.resize_hw,
        center_crop_hw=None,
        random_crop_hw=None,
        hflip_prob=cfg.train_hflip_prob,
        seed=cfg.seed,
    )
    return PairedTransforms(tcfg)


def _build_val_transforms(cfg: RealDataLoaderConfig) -> PairedTransforms | None:
    if cfg.resize_hw is None:
        return None
    tcfg = PairedTransformConfig(
        resize_hw=cfg.resize_hw,
        center_crop_hw=None,
        random_crop_hw=None,
        hflip_prob=0.0,
        seed=cfg.seed,
    )
    return PairedTransforms(tcfg)


def build_train_val_loaders(stage: str) -> tuple[DataLoader, DataLoader]:
    """External loader entrypoint consumed by training scripts.

    Config is controlled by AUTOHDR_* env vars so stage scripts can call this with only `stage`.
    """
    _ = stage  # stage-specific sampling can be added later without changing signature.
    cfg = resolve_real_data_config()
    _ensure_split_csvs(cfg)

    train_ds = PairedLensDataset(cfg.train_csv, transforms=_build_train_transforms(cfg))
    val_ds = PairedLensDataset(cfg.val_csv, transforms=_build_val_transforms(cfg))

    common_loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": bool(cfg.pin_memory and torch.cuda.is_available()),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader

