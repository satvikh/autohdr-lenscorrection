"""Image decode and dimension consistency checks."""

from __future__ import annotations

from pathlib import Path

from src.qa._image_utils import load_image_rgb_float
from src.qa._utils import _allowed_ext, _iter_image_files, _normalize_required_ids


def _find_image_for_id(directory: Path, image_id: str, allowed_ext) -> Path | None:
    for ext in allowed_ext:
        candidate = directory / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _resolve_gt_path(image_id: str, gt_root, gt_map) -> Path | None:
    if isinstance(gt_map, dict):
        mapped = gt_map.get(image_id)
        if mapped is None:
            return None
        return Path(mapped)
    if gt_root is None:
        return None
    return Path(gt_root) / f"{image_id}.png"


def check_images(pred_dir, required_ids, gt_root=None, config=None, gt_map=None) -> dict:
    """Validate required prediction images for decode and optional size consistency.

    Returns:
        {
          "ok": bool,
          "bad_files": [{"image_id": str, "reason": str}],
        }
    """
    pred_path = Path(pred_dir)
    required = _normalize_required_ids(required_ids)
    allowed_ext = _allowed_ext(config)
    cfg = config or {}
    image_cfg = cfg.get("image", {}) if isinstance(cfg, dict) else {}
    require_same_size = bool(image_cfg.get("require_same_size", True))
    bad_files: list[dict[str, str]] = []

    for image_id in required:
        pred_file = _find_image_for_id(pred_path, image_id, allowed_ext)
        if pred_file is None:
            bad_files.append(
                {
                    "image_id": image_id,
                    "reason": f"missing_prediction: no file for '{image_id}' in {pred_path}",
                }
            )
            continue
        try:
            pred_arr = load_image_rgb_float(pred_file)
        except ValueError as exc:
            bad_files.append({"image_id": image_id, "reason": f"pred_load_error: {exc}"})
            continue

        if not require_same_size:
            continue

        gt_path = _resolve_gt_path(image_id, gt_root=gt_root, gt_map=gt_map)
        if gt_path is None or not gt_path.exists():
            bad_files.append(
                {
                    "image_id": image_id,
                    "reason": f"missing_gt: expected GT for '{image_id}' (gt_root={gt_root})",
                }
            )
            continue
        try:
            gt_arr = load_image_rgb_float(gt_path)
        except ValueError as exc:
            bad_files.append({"image_id": image_id, "reason": f"gt_load_error: {exc}"})
            continue

        pred_size = (int(pred_arr.shape[1]), int(pred_arr.shape[0]))
        gt_size = (int(gt_arr.shape[1]), int(gt_arr.shape[0]))
        if pred_size != gt_size:
            bad_files.append(
                {
                    "image_id": image_id,
                    "reason": f"size_mismatch: pred={pred_size}, gt={gt_size}, pred_file={pred_file}, gt_file={gt_path}",
                }
            )

    # Include any decodable extras when strict full-folder scan is desired.
    if bool(cfg.get("scan_all_files", False)):
        for file in _iter_image_files(pred_path, allowed_ext):
            if file.stem in set(required):
                continue
            try:
                _ = load_image_rgb_float(file)
            except ValueError as exc:
                bad_files.append({"image_id": file.stem, "reason": f"pred_load_error: {exc}"})

    return {
        "ok": len(bad_files) == 0,
        "bad_files": bad_files,
    }
