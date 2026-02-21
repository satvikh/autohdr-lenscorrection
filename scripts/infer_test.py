from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Param-only baseline inference over a directory of images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {input_dir}")

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
    safety_fail_count = 0
    reason_counts: Counter[str] = Counter()

    for image_path in images:
        output_tensor, metadata = predictor.predict(image_path)
        out_path = output_dir / f"{image_path.stem}.jpg"
        h = int(output_tensor.shape[-2])
        w = int(output_tensor.shape[-1])
        save_jpeg(output_tensor, out_path, expected_hw=(h, w))

        processed += 1
        mode = str(metadata.get("mode_used", "unknown"))
        mode_counts[mode] += 1

        safety = metadata.get("safety", {})
        safe = bool(safety.get("safe", False))
        if not safe:
            safety_fail_count += 1
            for reason in safety.get("reasons", []):
                reason_counts[str(reason)] += 1

        print(f"[{processed}/{total}] {image_path.name} -> {out_path.name} mode={mode} safe={safe}")

    print("Summary")
    print(f"- processed: {processed}")
    print(f"- mode_used_counts: {dict(mode_counts)}")
    print(f"- safety_failures: {safety_fail_count}")
    print("- safety_reasons_top:")
    for reason, count in reason_counts.most_common(5):
        print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
