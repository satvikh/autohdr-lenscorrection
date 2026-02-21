from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import pandas as pd


def parse_exts(exts: str) -> List[str]:
    parsed = [e.strip().lower() for e in exts.split(",") if e.strip()]
    if not parsed:
        raise ValueError("No valid extensions provided.")
    return [e if e.startswith(".") else f".{e}" for e in parsed]


def scan_by_stem(directory: Path, exts: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        mapping[path.stem] = path
    return mapping


def edge_density(gray: np.ndarray) -> tuple[np.ndarray, float]:
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    density = float(np.count_nonzero(edges)) / float(edges.size) if edges.size else 0.0
    return edges, density


def hough_line_stats(edges: np.ndarray) -> tuple[int, float]:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=20,
        maxLineGap=5,
    )
    if lines is None:
        return 0, 0.0

    total_len = 0.0
    count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        total_len += float(np.hypot(x2 - x1, y2 - y1))
        count += 1
    return count, total_len


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_pair_row(image_id: str, input_path: Path, target_path: Path) -> dict:
    input_img = read_rgb(input_path)
    target_img = read_rgb(target_path)

    input_h, input_w = input_img.shape[:2]
    target_h, target_w = target_img.shape[:2]

    if (target_h, target_w) != (input_h, input_w):
        target_for_mae = cv2.resize(target_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    else:
        target_for_mae = target_img

    mae = float(np.mean(np.abs(input_img.astype(np.float32) - target_for_mae.astype(np.float32))) / 255.0)

    input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

    _, edge_density_input = edge_density(input_gray)
    target_edges, edge_density_target = edge_density(target_gray)
    line_count, line_total_len = hough_line_stats(target_edges)

    return {
        "image_id": image_id,
        "input_path": str(input_path),
        "target_path": str(target_path),
        "input_h": int(input_h),
        "input_w": int(input_w),
        "target_h": int(target_h),
        "target_w": int(target_w),
        "aspect_input": float(input_w) / float(input_h) if input_h else 0.0,
        "aspect_target": float(target_w) / float(target_h) if target_h else 0.0,
        "mae_input_target": mae,
        "edge_density_input": float(edge_density_input),
        "edge_density_target": float(edge_density_target),
        "line_count": int(line_count),
        "line_total_len": float(line_total_len),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paired lens-correction dataset.")
    parser.add_argument("--original_dir", required=True, type=Path)
    parser.add_argument("--generated_dir", required=True, type=Path)
    parser.add_argument("--out_csv", default=Path("data/metadata/dataset_audit.csv"), type=Path)
    parser.add_argument("--exts", default=".jpg,.png,.jpeg", type=str)
    args = parser.parse_args()

    exts = parse_exts(args.exts)
    original_map = scan_by_stem(args.original_dir, exts)
    generated_map = scan_by_stem(args.generated_dir, exts)
    common_ids = sorted(set(original_map).intersection(generated_map))

    rows = [compute_pair_row(image_id, original_map[image_id], generated_map[image_id]) for image_id in common_ids]

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
