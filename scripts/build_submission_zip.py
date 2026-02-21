"""Build submission zip with QA validation gates."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qa.filename_check import check_filenames
from src.qa.image_integrity_check import check_images


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    try:
        data = ast.literal_eval(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise ValueError(f"Failed to parse config file '{path}': {exc}") from exc


def _allowed_ext(config: dict) -> tuple[str, ...]:
    image_cfg = config.get("image", {}) if isinstance(config, dict) else {}
    exts = image_cfg.get("allowed_ext", [".png", ".jpg", ".jpeg"]) if isinstance(image_cfg, dict) else [".png", ".jpg", ".jpeg"]
    out = []
    for ext in exts:
        s = str(ext).lower()
        out.append(s if s.startswith(".") else f".{s}")
    return tuple(out)


def _load_ids_from_split(split_csv: Path) -> list[str]:
    ids = []
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise ValueError("split_csv must contain 'image_id' column")
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                ids.append(image_id)
    return sorted(set(ids))


def _load_ids_from_file(ids_file: Path) -> list[str]:
    text = ids_file.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if ids_file.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return sorted({str(x) for x in data})
        if isinstance(data, dict) and isinstance(data.get("ids"), list):
            return sorted({str(x) for x in data["ids"]})

    if ids_file.suffix.lower() == ".csv":
        ids = []
        with ids_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "image_id" in reader.fieldnames:
                for row in reader:
                    image_id = str(row.get("image_id", "")).strip()
                    if image_id:
                        ids.append(image_id)
        if ids:
            return sorted(set(ids))

    return sorted({line.strip() for line in text.splitlines() if line.strip()})


def _find_pred_file(pred_dir: Path, image_id: str, allowed_ext: tuple[str, ...]) -> Path | None:
    for ext in allowed_ext:
        p = pred_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for submission zip creation."""
    parser = argparse.ArgumentParser(description="Build submission zip after QA checks.")
    parser.add_argument("--pred_dir", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split_csv")
    group.add_argument("--ids_file")
    parser.add_argument("--out_zip", default="submission.zip")
    parser.add_argument("--config", default=None)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--force_zip", action="store_true", default=False)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Run QA checks, write QA report, and build submission zip when allowed."""
    config = _load_config(args.config)
    pred_dir = Path(args.pred_dir)

    if args.split_csv:
        required_ids = _load_ids_from_split(Path(args.split_csv))
    else:
        required_ids = _load_ids_from_file(Path(args.ids_file))

    filename_res = check_filenames(pred_dir, required_ids, config)
    # No gt_root/gt_map expected for submission packaging; decode checks still run.
    image_res = check_images(
        pred_dir,
        required_ids,
        gt_root=None,
        config={**config, "image": {**config.get("image", {}), "require_same_size": False}},
        gt_map=None,
    )

    qa_ok = bool(filename_res.get("ok", False) and image_res.get("ok", False))
    qa_report = {
        "ok": qa_ok,
        "filename_check": filename_res,
        "image_integrity_check": image_res,
        "strict": bool(args.strict),
        "force_zip": bool(args.force_zip),
        "required_count": len(required_ids),
    }

    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    qa_report_path = out_zip.with_name("submission_qa.json")
    qa_report_path.write_text(json.dumps(qa_report, indent=2, allow_nan=True), encoding="utf-8")

    if args.strict and not qa_ok and not args.force_zip:
        return 2

    allowed_ext = _allowed_ext(config)
    zip_subdir = str(config.get("submission", {}).get("zip_subdir", "")).strip()

    files_to_zip: list[tuple[Path, str]] = []
    for image_id in required_ids:
        pred_path = _find_pred_file(pred_dir, image_id, allowed_ext)
        if pred_path is None:
            continue
        arcname = pred_path.name
        if zip_subdir:
            arcname = f"{zip_subdir.rstrip('/')}/{arcname}"
        files_to_zip.append((pred_path, arcname))

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src, arcname in sorted(files_to_zip, key=lambda x: x[1]):
            zf.write(src, arcname=arcname)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint returning process-like exit code."""
    try:
        args = parse_args(argv)
        return run(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"build_submission_zip failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
