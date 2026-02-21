from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.proxy_score import compute_proxy_score
from src.models.hybrid_model import HybridLensCorrectionModel
from src.train.config_loader import load_model_config
from src.train.proxy_hooks import compute_proxy_metrics_for_batch
from src.train.stage_configs import get_stage_toggles
from src.train.warp_backends import Person1GeometryWarpBackend


def _agg_init() -> dict[str, list[float]]:
    return {
        "total": [],
        "edge": [],
        "line": [],
        "grad": [],
        "ssim": [],
        "mae": [],
        "hard_fail": [],
    }


def _agg_update(acc: dict[str, list[float]], metrics: dict[str, float]) -> None:
    for src, dst in (
        ("proxy_total_score", "total"),
        ("proxy_edge", "edge"),
        ("proxy_line", "line"),
        ("proxy_grad", "grad"),
        ("proxy_ssim", "ssim"),
        ("proxy_mae", "mae"),
        ("proxy_hard_fail", "hard_fail"),
    ):
        if src in metrics:
            acc[dst].append(float(metrics[src]))


def _agg_mean(acc: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, values in acc.items():
        if not values:
            out[k] = float("nan")
        else:
            out[k] = float(sum(values) / len(values))
    return out


def _run_proxy(pred: torch.Tensor, target: torch.Tensor, proxy_config: dict[str, Any]) -> dict[str, float]:
    return compute_proxy_metrics_for_batch(
        scorer=compute_proxy_score,
        pred_batch=pred,
        target_batch=target,
        config=proxy_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare identity baseline vs model predictions on same subset.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", type=str, default="stage1_param_only")
    parser.add_argument("--subset", choices=["train", "val"], default="val")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--proxy-config-json", type=Path, default=None)
    parser.add_argument("--loader-module", type=str, default="src.data.real_loader")
    parser.add_argument("--loader-fn", type=str, default="build_train_val_loaders")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    model_cfg, bounds = load_model_config(args.model_config)
    model = HybridLensCorrectionModel(config=model_cfg, param_bounds=bounds)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    requested = str(args.device).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable. Falling back to CPU.")
        requested = "cpu"
    device = torch.device(requested)
    model.to(device)
    model.eval()

    stage = get_stage_toggles(args.stage)
    backend = Person1GeometryWarpBackend()

    proxy_config: dict[str, Any] = {}
    if args.proxy_config_json is not None:
        proxy_config = json.loads(args.proxy_config_json.read_text(encoding="utf-8"))

    module = __import__(args.loader_module, fromlist=[args.loader_fn])
    loader_builder = getattr(module, args.loader_fn)
    train_loader, val_loader = loader_builder(stage=stage.name)
    loader = train_loader if args.subset == "train" else val_loader

    identity_acc = _agg_init()
    model_acc = _agg_init()
    sample_count = 0
    batches_seen = 0

    with torch.no_grad():
        for batch in loader:
            batches_seen += 1
            if args.max_batches > 0 and batches_seen > args.max_batches:
                break

            input_image = batch["input_image"].to(device=device, dtype=torch.float32)
            target_image = batch["target_image"].to(device=device, dtype=torch.float32)

            raw_out = model(input_image)
            residual = raw_out.get("residual_flow_lowres") if stage.use_residual else None
            warp_out = backend.warp(input_image, raw_out["params"], residual)
            pred_image = warp_out["pred_image"]

            id_metrics = _run_proxy(input_image, target_image, proxy_config=proxy_config)
            model_metrics = _run_proxy(pred_image, target_image, proxy_config=proxy_config)
            _agg_update(identity_acc, id_metrics)
            _agg_update(model_acc, model_metrics)
            sample_count += int(input_image.shape[0])

    identity_mean = _agg_mean(identity_acc)
    model_mean = _agg_mean(model_acc)
    delta = {k: float(model_mean[k] - identity_mean[k]) for k in identity_mean.keys()}

    summary = {
        "stage": stage.name,
        "subset": args.subset,
        "sample_count": sample_count,
        "batches_seen": batches_seen,
        "identity": identity_mean,
        "model": model_mean,
        "delta_model_minus_identity": delta,
        "checkpoint": str(args.checkpoint),
    }

    print(json.dumps(summary, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote: {args.json_out}")


if __name__ == "__main__":
    main()
