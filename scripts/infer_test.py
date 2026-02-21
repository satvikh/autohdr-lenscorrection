from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor
from src.inference.writer import save_jpeg


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class NeutralModel:
    """Default stub model that emits neutral param-only warp (identity)."""

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        params = torch.zeros((b, 8), dtype=image.dtype, device=image.device)
        params[:, 7] = 1.0
        return {"params": params}


def _collect_images(input_dir: Path) -> list[Path]:
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference over a directory of images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint-id", type=str, default="neutral_stub")
    parser.add_argument("--config-path", type=Path, default=None)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    checkpoint_id: str = args.checkpoint_id
    config_path: Path | None = args.config_path

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {input_dir}")
    if config_path is not None and (not config_path.exists() or not config_path.is_file()):
        raise SystemExit(f"config_path is not a file: {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = NeutralModel()
    predictor = Predictor(model=model)

    images = _collect_images(input_dir)
    total = len(images)

    if total == 0:
        print("No input images found.")
        return

    processed = 0
    mode_counts: Counter[str] = Counter()
    unsafe_trigger_count = 0
    fallback_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    residual_metric_sums = {
        "residual_dx_abs_mean_norm": 0.0,
        "residual_dy_abs_mean_norm": 0.0,
        "residual_dx_abs_max_norm": 0.0,
        "residual_dy_abs_max_norm": 0.0,
    }
    residual_metric_count = 0

    for image_path in images:
        output_tensor, metadata = predictor.predict(image_path)
        out_path = output_dir / f"{image_path.stem}.jpg"
        h = int(output_tensor.shape[-2])
        w = int(output_tensor.shape[-1])
        save_jpeg(output_tensor, out_path, expected_hw=(h, w))

        processed += 1
        mode = str(metadata.get("mode_used", "unknown"))
        mode_counts[mode] += 1

        initial_safety = metadata.get("initial_safety", {})
        initial_safe = bool(initial_safety.get("safe", True))
        if not initial_safe:
            unsafe_trigger_count += 1
            for reason in initial_safety.get("reasons", []):
                reason_counts[str(reason)] += 1

        if not initial_safe:
            fallback_counts[mode] += 1

        initial_metrics = initial_safety.get("metrics", {})
        if all(k in initial_metrics for k in residual_metric_sums):
            has_residual_signal = any(abs(float(initial_metrics[k])) > 0.0 for k in residual_metric_sums)
            if has_residual_signal:
                residual_metric_count += 1
                for k in residual_metric_sums:
                    residual_metric_sums[k] += float(initial_metrics[k])

        final_safe = bool(metadata.get("safety", {}).get("safe", False))
        print(
            f"[{processed}/{total}] {image_path.name} -> {out_path.name} "
            f"mode={mode} initial_safe={initial_safe} final_safe={final_safe}"
        )

    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    config_hash = "none"
    if config_path is not None:
        config_hash = _sha256_file(config_path)

    run_metadata = {
        "checkpoint_id": checkpoint_id,
        "config_hash": config_hash,
        "timestamp_utc": run_timestamp_utc,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed": processed,
        "mode_counts": dict(mode_counts),
        "unsafe_trigger_count": unsafe_trigger_count,
        "fallback_counts": dict(fallback_counts),
        "safety_reason_counts": dict(reason_counts),
    }
    if residual_metric_count > 0:
        run_metadata["residual_metrics_avg"] = {
            key: (total_val / residual_metric_count) for key, total_val in residual_metric_sums.items()
        }

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")

    print("Summary")
    print(f"- processed: {processed}")
    print(f"- mode_used_counts: {dict(mode_counts)}")
    print(f"- unsafe_triggers: {unsafe_trigger_count}")
    print(f"- fallback_counts: {dict(fallback_counts)}")
    print("- safety_reasons_top:")
    for reason, count in reason_counts.most_common(5):
        print(f"  - {reason}: {count}")

    if residual_metric_count > 0:
        print("- residual_metrics_avg:")
        for key, total_val in residual_metric_sums.items():
            print(f"  - {key}: {total_val / residual_metric_count:.6f}")
    print(f"- run_metadata: {metadata_path}")


if __name__ == "__main__":
    main()
