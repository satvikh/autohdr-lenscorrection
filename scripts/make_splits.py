from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_COLUMNS = ["image_id", "input_path", "target_path"]


def quantile_bins(series: pd.Series, bins: int, prefix: str) -> pd.Series:
    if bins <= 1:
        return pd.Series([f"{prefix}0"] * len(series), index=series.index)
    binned = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    binned = binned.fillna(0).astype(int)
    return binned.map(lambda x: f"{prefix}{x}")


def stratified_split(frame: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    val_indices = []
    train_indices = []

    for _, group in frame.groupby("stratum", sort=True):
        idx = group.index.to_numpy()
        idx = rng.permutation(idx)
        n = len(idx)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(n * val_frac))
            if val_frac > 0 and n_val == 0:
                n_val = 1
            n_val = min(n - 1, n_val)

        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())

    train_df = frame.loc[sorted(train_indices)].copy()
    val_df = frame.loc[sorted(val_indices)].copy()
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val splits from dataset audit CSV.")
    parser.add_argument("--audit_csv", default=Path("data/metadata/dataset_audit.csv"), type=Path)
    parser.add_argument("--out_dir", default=Path("data/splits"), type=Path)
    parser.add_argument("--val_frac", default=0.2, type=float)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--bins_res", default=3, type=int)
    parser.add_argument("--bins_line", default=3, type=int)
    args = parser.parse_args()

    if not (0.0 <= args.val_frac < 1.0):
        raise ValueError("val_frac must be in [0, 1).")

    frame = pd.read_csv(args.audit_csv)
    required_cols = {"image_id", "input_path", "target_path", "target_h", "target_w", "line_total_len"}
    missing = required_cols.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in audit CSV: {sorted(missing)}")

    frame = frame.copy()
    frame["target_pixels"] = frame["target_h"].astype(float) * frame["target_w"].astype(float)
    frame["res_bin"] = quantile_bins(frame["target_pixels"], args.bins_res, prefix="r")
    frame["line_bin"] = quantile_bins(frame["line_total_len"].astype(float), args.bins_line, prefix="l")
    frame["stratum"] = frame["res_bin"].astype(str) + "|" + frame["line_bin"].astype(str)

    train_df, val_df = stratified_split(frame, val_frac=args.val_frac, seed=args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / "train_split.csv"
    val_out = out_dir / "val_split.csv"
    train_df.loc[:, SPLIT_COLUMNS].to_csv(train_out, index=False)
    val_df.loc[:, SPLIT_COLUMNS].to_csv(val_out, index=False)

    full_counts = frame.groupby("stratum").size().rename("total_count")
    train_counts = train_df.groupby("stratum").size().rename("train_count")
    val_counts = val_df.groupby("stratum").size().rename("val_count")

    report = pd.concat([full_counts, train_counts, val_counts], axis=1).fillna(0).reset_index()
    report["train_count"] = report["train_count"].astype(int)
    report["val_count"] = report["val_count"].astype(int)
    report_out = out_dir / "split_strata_report.csv"
    report.to_csv(report_out, index=False)

    print(f"Wrote train split: {train_out} ({len(train_df)} rows)")
    print(f"Wrote val split: {val_out} ({len(val_df)} rows)")
    print(f"Wrote strata report: {report_out}")


if __name__ == "__main__":
    main()
