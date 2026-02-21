"""Internal helpers for QA modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def _normalize_required_ids(required_ids: Iterable[str]) -> list[str]:
    return sorted({str(x) for x in required_ids})


def _allowed_ext(config) -> Sequence[str]:
    cfg = config or {}
    exts = [".jpg", ".jpeg", ".png"]
    if isinstance(cfg, dict):
        image_cfg = cfg.get("image", {})
        if isinstance(image_cfg, dict) and image_cfg.get("allowed_ext") is not None:
            exts = image_cfg.get("allowed_ext")
        elif cfg.get("allowed_ext") is not None:
            exts = cfg.get("allowed_ext")
    normalized = []
    for ext in exts:
        s = str(ext).lower()
        normalized.append(s if s.startswith(".") else f".{s}")
    return tuple(normalized)


def _iter_image_files(pred_dir: str | Path, allowed_ext: Sequence[str]):
    p = Path(pred_dir)
    for item in sorted(p.iterdir()):
        if item.is_file() and item.suffix.lower() in allowed_ext:
            yield item
