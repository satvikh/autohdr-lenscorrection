"""CLI for proxy-score validation against a split CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.proxy_score import aggregate_scores, compute_proxy_score
from src.metrics.proxy_ssim_mae import compute_mae


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    # Try YAML first when available.
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

    # JSON fallback.
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Python dict literal fallback.
    try:
        data = ast.literal_eval(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise ValueError(f"Failed to parse config file '{path}': {exc}") from exc


def _allowed_ext(config: dict) -> tuple[str, ...]:
    image_cfg = config.get("image", {}) if isinstance(config, dict) else {}
    exts = image_cfg.get("allowed_ext", [".png", ".jpg", ".jpeg"]) if isinstance(image_cfg, dict) else [".png", ".jpg", ".jpeg"]
    normalized = []
    for ext in exts:
        s = str(ext).lower()
        normalized.append(s if s.startswith(".") else f".{s}")
    return tuple(normalized)


def _find_pred_path(pred_dir: Path, image_id: str, allowed_ext: tuple[str, ...]) -> Path | None:
    for ext in allowed_ext:
        p = pred_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def _build_gt_map(split_csv: Path, gt_root: str | None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("split_csv has no header")
        headers = set(reader.fieldnames)
        if "image_id" not in headers:
            raise ValueError("split_csv must contain 'image_id' column")
        if "target_path" not in headers and "rel_target_path" not in headers:
            raise ValueError("split_csv must contain either 'target_path' or 'rel_target_path'")

        split_parent = split_csv.parent
        gt_root_path = Path(gt_root) if gt_root else None
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                continue

            if row.get("target_path"):
                tp = Path(str(row["target_path"]).strip())
                if not tp.is_absolute():
                    tp = split_parent / tp
                mapping[image_id] = tp
                continue

            rel_target = str(row.get("rel_target_path", "")).strip()
            if not rel_target:
                raise ValueError(f"Missing target path for image_id={image_id}")
            if gt_root_path is None:
                raise ValueError("--gt_root is required when split_csv uses rel_target_path")
            mapping[image_id] = gt_root_path / rel_target

    return mapping


def _hardfail_total(config: dict) -> float:
    fail_policy = str(config.get("aggregation", {}).get("fail_policy", "exclude"))
    if fail_policy == "score_zero":
        return 0.0
    if fail_policy == "score_neg_inf":
        return -1e9
    return float("nan")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for proxy validation."""
    parser = argparse.ArgumentParser(description="Validate proxy score over a split CSV.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--gt_root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out_dir", default="reports/validate_proxy")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute validation run and emit per-image + summary reports."""
    config = _load_config(args.config)
    allowed_ext = _allowed_ext(config)

    pred_dir = Path(args.pred_dir)
    split_csv = Path(args.split_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_map = _build_gt_map(split_csv, args.gt_root)

    rows: list[dict] = []
    per_image_records: list[dict] = []

    for image_id in sorted(gt_map.keys()):
        gt_path = gt_map[image_id]
        pred_path = _find_pred_path(pred_dir, image_id, allowed_ext)

        if pred_path is None or not gt_path.exists():
            reasons = []
            if pred_path is None:
                reasons.append("missing_prediction")
            if not gt_path.exists():
                reasons.append("missing_gt")
            score = {
                "total": _hardfail_total(config),
                "subscores": {"edge": float("nan"), "line": float("nan"), "grad": float("nan"), "ssim": float("nan"), "mae": float("nan")},
                "flags": {"hardfail": True, "reasons": reasons},
            }
            mae_raw = float("nan")
            mae_score = float("nan")
        else:
            try:
                score = compute_proxy_score(pred_path, gt_path, config)
            except Exception as exc:
                score = {
                    "total": _hardfail_total(config),
                    "subscores": {"edge": float("nan"), "line": float("nan"), "grad": float("nan"), "ssim": float("nan"), "mae": float("nan")},
                    "flags": {"hardfail": True, "reasons": [f"score_error:{exc}"]},
                }
            try:
                mae_raw = float(compute_mae(pred_path, gt_path, config))
                mae_score = float(max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, mae_raw)))))
            except Exception:
                mae_raw = float("nan")
                mae_score = float("nan")

        row = {"image_id": image_id, **score}
        rows.append(row)

        subs = score.get("subscores", {})
        per_image_records.append(
            {
                "image_id": image_id,
                "total": score.get("total", float("nan")),
                "hardfail": bool(score.get("flags", {}).get("hardfail", False)),
                "reasons": "|".join(str(x) for x in score.get("flags", {}).get("reasons", [])),
                "edge": subs.get("edge", float("nan")),
                "ssim": subs.get("ssim", float("nan")),
                "mae": mae_raw,
                "mae_score": mae_score,
            }
        )

    summary = aggregate_scores(rows, config)

    per_image_csv = out_dir / "per_image_scores.csv"
    with per_image_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["image_id", "total", "hardfail", "reasons", "edge", "ssim", "mae", "mae_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_records)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    fail_count = int(summary.get("fail_count", 0))
    total_count = int(summary.get("count", 0))
    fail_rate = float(fail_count / total_count) if total_count > 0 else 0.0
    allowed_fail_rate = float(config.get("aggregation", {}).get("allowed_fail_rate", 0.0))

    if fail_rate <= allowed_fail_rate:
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint returning process-like exit code."""
    try:
        args = parse_args(argv)
        return run(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"validate_proxy failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
