"""Filename validation for submission outputs."""

from __future__ import annotations

from pathlib import Path

from src.qa._utils import _allowed_ext, _normalize_required_ids


def check_filenames(pred_dir, required_ids, config) -> dict:
    """Validate filename coverage against required ids and allowed extensions."""
    pred_path = Path(pred_dir)
    required = _normalize_required_ids(required_ids)
    required_set = set(required)
    allowed_ext = _allowed_ext(config)
    cfg = config or {}
    qa_cfg = cfg.get("qa", {}) if isinstance(cfg, dict) else {}
    allow_extra_files = bool(qa_cfg.get("allow_extra_files", False))

    files = [p for p in sorted(pred_path.iterdir()) if p.is_file()]

    valid_by_id: dict[str, list[str]] = {}
    bad_names: list[str] = []
    extra: list[str] = []

    for file in files:
        stem = file.stem
        ext = file.suffix.lower()
        filename = file.name
        if ext not in allowed_ext:
            bad_names.append(filename)
            continue
        if stem not in required_set:
            if not allow_extra_files:
                extra.append(filename)
            continue
        valid_by_id.setdefault(stem, []).append(filename)

    missing = sorted([image_id for image_id in required if image_id not in valid_by_id])
    for image_id, names in valid_by_id.items():
        if len(names) > 1:
            bad_names.extend(sorted(names))

    ok = len(missing) == 0 and len(extra) == 0 and len(bad_names) == 0
    return {
        "ok": ok,
        "missing": missing,
        "extra": sorted(extra),
        "bad_names": sorted(bad_names),
    }
