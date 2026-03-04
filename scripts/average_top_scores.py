"""Compute average score for top-N rows in a submission CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_TOP_K = (1000, 500, 250, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average score for top-N images.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="submission.csv",
        help="Path to submission CSV (default: submission.csv)",
    )
    return parser.parse_args()


def load_scores(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    scores: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        if "score" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'score' column.")

        for row in reader:
            raw = str(row.get("score", "")).strip()
            if not raw:
                continue
            try:
                scores.append(float(raw))
            except ValueError as exc:
                raise ValueError(f"Invalid score value: {raw}") from exc

    if not scores:
        raise ValueError("No valid scores found in CSV.")

    return scores


def compute_top_k_averages(scores: list[float], top_k: tuple[int, ...] = DEFAULT_TOP_K) -> list[tuple[int, int, float]]:
    sorted_scores = sorted(scores, reverse=True)
    results: list[tuple[int, int, float]] = []
    for k in top_k:
        used = min(k, len(sorted_scores))
        avg = sum(sorted_scores[:used]) / used
        results.append((k, used, avg))
    return results


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    scores = load_scores(csv_path)
    results = compute_top_k_averages(scores)

    print(f"Loaded {len(scores)} scores from {csv_path}")
    for requested_k, used_k, avg in results:
        print(f"Top {requested_k:>4} (used {used_k:>4}) average: {avg:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
