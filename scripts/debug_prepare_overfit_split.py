from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random


def _scan_pairs(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for orig_path in sorted(root.glob("*_original.jpg")):
        pair_id = orig_path.name[: -len("_original.jpg")]
        gen_path = root / f"{pair_id}_generated.jpg"
        if not gen_path.exists():
            continue
        rows.append(
            {
                "image_id": pair_id,
                "input_path": str(orig_path),
                "target_path": str(gen_path),
            }
        )
    return rows


def _write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "input_path", "target_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an overfit split where train and val use the same subset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("automatic-lens-correction/lens-correction-train-cleaned"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits/debug_overfit"))
    parser.add_argument("--num-pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive.")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    rows = _scan_pairs(args.data_root)
    if not rows:
        raise ValueError(f"No training pairs found in: {args.data_root}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    subset = rows[: args.num_pairs]
    if len(subset) < args.num_pairs:
        print(f"[warn] Requested {args.num_pairs} pairs, found only {len(subset)}.")

    train_csv = args.out_dir / "train_split.csv"
    val_csv = args.out_dir / "val_split.csv"
    _write_split_csv(train_csv, subset)
    _write_split_csv(val_csv, subset)

    report = {
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "num_pairs_requested": int(args.num_pairs),
        "num_pairs_written": int(len(subset)),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "seed": int(args.seed),
        "shuffle": bool(args.shuffle),
        "note": "train and val intentionally identical for micro-overfit debug",
    }
    (args.out_dir / "overfit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote overfit train split: {train_csv} ({len(subset)} rows)")
    print(f"Wrote overfit val split: {val_csv} ({len(subset)} rows)")
    print(f"Wrote report: {args.out_dir / 'overfit_report.json'}")


if __name__ == "__main__":
    main()
