from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer

from src.geometry.coords import make_identity_grid
from src.losses.composite import CompositeLoss
from src.train.amp_utils import autocast_context
from src.train.protocols import WarpBackend
from src.train.stage_configs import StageToggles


@dataclass
class TrainStepResult:
    total_loss: Tensor
    components: dict[str, Tensor]
    pred_image: Tensor
    model_outputs: dict[str, Any]
    warp_outputs: dict[str, Any]
    diagnostics: dict[str, float]


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _tensor_scalar(value: Tensor) -> float:
    return float(value.detach().item())


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _group_name(param_name: str) -> str:
    if param_name.startswith("backbone."):
        return "backbone"
    if param_name.startswith("param_head."):
        return "param_head"
    if param_name.startswith("residual_head."):
        return "residual_head"
    return "other"


def _group_norm_from_tensors(values_by_name: dict[str, Tensor]) -> dict[str, float]:
    sq_sum: dict[str, float] = {"backbone": 0.0, "param_head": 0.0, "residual_head": 0.0, "other": 0.0}
    for name, value in values_by_name.items():
        group = _group_name(name)
        safe_value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(torch.linalg.vector_norm(safe_value).item())
        sq_sum[group] += norm * norm
    return {f"{k}": math.sqrt(v) for k, v in sq_sum.items()}


def _collect_group_grad_norms(model: nn.Module) -> dict[str, float]:
    grads: dict[str, Tensor] = {}
    nonfinite: dict[str, float] = {"backbone": 0.0, "param_head": 0.0, "residual_head": 0.0, "other": 0.0}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grad = p.grad.detach()
        grads[name] = grad
        if not torch.isfinite(grad).all():
            nonfinite[_group_name(name)] += 1.0
    norms = _group_norm_from_tensors(grads)
    return {
        "grad_norm_backbone": norms["backbone"],
        "grad_norm_param_head": norms["param_head"],
        "grad_norm_residual_head": norms["residual_head"],
        "grad_norm_other": norms["other"],
        "grad_nonfinite_count_backbone": nonfinite["backbone"],
        "grad_nonfinite_count_param_head": nonfinite["param_head"],
        "grad_nonfinite_count_residual_head": nonfinite["residual_head"],
        "grad_nonfinite_count_other": nonfinite["other"],
    }


def _collect_group_param_deltas(model: nn.Module, before: dict[str, Tensor]) -> dict[str, float]:
    deltas: dict[str, Tensor] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        prev = before.get(name)
        if prev is None:
            continue
        deltas[name] = p.detach() - prev
    norms = _group_norm_from_tensors(deltas)
    return {
        "param_delta_backbone": norms["backbone"],
        "param_delta_param_head": norms["param_head"],
        "param_delta_residual_head": norms["residual_head"],
        "param_delta_other": norms["other"],
    }


def _norm_delta_bhwc_to_px_stats(delta_bhwc: Tensor) -> dict[str, float]:
    if delta_bhwc.ndim != 4 or delta_bhwc.shape[-1] != 2:
        raise ValueError(f"Expected BHWC normalized delta with last dim 2, got {tuple(delta_bhwc.shape)}")
    h = int(delta_bhwc.shape[1])
    w = int(delta_bhwc.shape[2])
    sx = (w - 1) * 0.5 if w > 1 else 0.0
    sy = (h - 1) * 0.5 if h > 1 else 0.0

    dx = delta_bhwc[..., 0] * sx
    dy = delta_bhwc[..., 1] * sy
    mag = torch.sqrt(dx * dx + dy * dy)
    return {
        "mean": _tensor_scalar(mag.mean()),
        "max": _tensor_scalar(mag.amax()),
        "std": _tensor_scalar(mag.std(unbiased=False)),
        "dx_mean": _tensor_scalar(dx.mean()),
        "dy_mean": _tensor_scalar(dy.mean()),
        "abs_mean": _tensor_scalar(mag.abs().mean()),
    }


def _extract_param_bounds_and_names(model: nn.Module, n_params: int) -> tuple[list[str], list[tuple[float, float]]]:
    """Best-effort extraction of parameter names and bounds from model param head.

    Fallback names are `p0..pN-1` and bounds are `(-1, 1)` when unavailable.
    """
    default_names = [f"p{i}" for i in range(n_params)]
    default_bounds = [(-1.0, 1.0)] * n_params

    param_head = getattr(model, "param_head", None)
    if param_head is None:
        return default_names, default_bounds

    names_raw = getattr(param_head, "param_names", None)
    bounds_raw = getattr(param_head, "bounds", None)
    if not isinstance(names_raw, list) or bounds_raw is None:
        return default_names, default_bounds

    names: list[str] = []
    bounds: list[tuple[float, float]] = []
    for idx, name in enumerate(names_raw):
        if idx >= n_params:
            break
        pair = getattr(bounds_raw, str(name), None)
        if isinstance(pair, tuple) and len(pair) == 2:
            lo = float(pair[0])
            hi = float(pair[1])
            if hi > lo:
                names.append(str(name))
                bounds.append((lo, hi))
                continue
        names.append(str(name))
        bounds.append((-1.0, 1.0))

    if len(names) < n_params:
        for idx in range(len(names), n_params):
            names.append(f"p{idx}")
            bounds.append((-1.0, 1.0))
    return names, bounds


def _extract_model_diagnostics(raw_outputs: dict[str, Any], model: nn.Module) -> dict[str, float]:
    diag: dict[str, float] = {}

    params = raw_outputs.get("params")
    params_raw = raw_outputs.get("params_raw")
    if torch.is_tensor(params_raw):
        diag["params_raw_mean"] = _tensor_scalar(params_raw.mean())
        diag["params_raw_abs_mean"] = _tensor_scalar(params_raw.abs().mean())
        diag["params_raw_abs_max"] = _tensor_scalar(params_raw.abs().amax())
        diag["params_raw_std"] = _tensor_scalar(params_raw.std(unbiased=False))

    if torch.is_tensor(params):
        diag["params_mean"] = _tensor_scalar(params.mean())
        diag["params_abs_mean"] = _tensor_scalar(params.abs().mean())
        diag["params_abs_max"] = _tensor_scalar(params.abs().amax())
        diag["params_std"] = _tensor_scalar(params.std(unbiased=False))
        if params.shape[1] >= 1:
            diag["param_k1_mean"] = _tensor_scalar(params[:, 0].mean())
        if params.shape[1] >= 2:
            diag["param_k2_mean"] = _tensor_scalar(params[:, 1].mean())
        if params.shape[1] >= 3:
            diag["param_k3_mean"] = _tensor_scalar(params[:, 2].mean())
        if params.shape[1] >= 8:
            diag["param_scale_mean"] = _tensor_scalar(params[:, 7].mean())

        names, bounds = _extract_param_bounds_and_names(model, int(params.shape[1]))
        sat_max = 0.0
        for i, name in enumerate(names):
            lo, hi = bounds[i]
            margin = max((hi - lo) * 0.02, 1e-9)
            is_sat = (params[:, i] <= (lo + margin)) | (params[:, i] >= (hi - margin))
            sat_frac = _tensor_scalar(is_sat.float().mean())
            diag[f"param_sat_frac_{name}"] = sat_frac
            sat_max = max(sat_max, sat_frac)
        diag["param_sat_frac_max"] = sat_max

    residual_low = raw_outputs.get("residual_flow_lowres")
    if residual_low is None:
        residual_low = raw_outputs.get("residual_flow")
    if torch.is_tensor(residual_low):
        diag["residual_lowres_mean_px"] = _tensor_scalar(residual_low.mean())
        diag["residual_lowres_std_px"] = _tensor_scalar(residual_low.std(unbiased=False))
        diag["residual_lowres_abs_mean_px"] = _tensor_scalar(residual_low.abs().mean())
        diag["residual_lowres_abs_max_px"] = _tensor_scalar(residual_low.abs().amax())

    residual_full = raw_outputs.get("residual_flow_fullres")
    if torch.is_tensor(residual_full):
        diag["residual_fullres_mean_px"] = _tensor_scalar(residual_full.mean())
        diag["residual_fullres_std_px"] = _tensor_scalar(residual_full.std(unbiased=False))
        diag["residual_fullres_abs_mean_px"] = _tensor_scalar(residual_full.abs().mean())
        diag["residual_fullres_abs_max_px"] = _tensor_scalar(residual_full.abs().amax())

    debug_stats = raw_outputs.get("debug_stats")
    if isinstance(debug_stats, dict):
        for k, v in debug_stats.items():
            if torch.is_tensor(v) and v.numel() == 1:
                diag[f"model_debug_{k}"] = _tensor_scalar(v)

    return diag


def _extract_warp_diagnostics(warp_outputs: dict[str, Any]) -> dict[str, float]:
    diag: dict[str, float] = {}
    warp_stats = warp_outputs.get("warp_stats")

    if isinstance(warp_stats, dict):
        for k, v in warp_stats.items():
            if isinstance(v, (int, float)):
                diag[f"warp_{k}"] = float(v)
            elif torch.is_tensor(v) and v.numel() == 1:
                diag[f"warp_{k}"] = _tensor_scalar(v)
            elif k == "safety" and isinstance(v, dict):
                diag["warp_safety_safe"] = 1.0 if bool(v.get("safe", False)) else 0.0

    residual_norm = warp_outputs.get("residual_flow_fullres_norm")
    if torch.is_tensor(residual_norm):
        diag["residual_fullres_norm_abs_mean"] = _tensor_scalar(residual_norm.abs().mean())
        diag["residual_fullres_norm_abs_max"] = _tensor_scalar(residual_norm.abs().amax())
        residual_stats = _norm_delta_bhwc_to_px_stats(residual_norm)
        for k, v in residual_stats.items():
            diag[f"residual_fullres_norm_disp_px_{k}"] = v

    param_grid = warp_outputs.get("param_grid")
    final_grid = warp_outputs.get("final_grid")
    if torch.is_tensor(param_grid) and param_grid.ndim == 4 and param_grid.shape[-1] == 2:
        b, h, w, _ = param_grid.shape
        identity = make_identity_grid(
            int(b),
            int(h),
            int(w),
            device=param_grid.device,
            dtype=param_grid.dtype,
        )
        param_delta = param_grid - identity
        pstats = _norm_delta_bhwc_to_px_stats(param_delta)
        for k, v in pstats.items():
            diag[f"param_grid_disp_px_{k}"] = v

        if torch.is_tensor(final_grid) and final_grid.shape == param_grid.shape:
            final_delta = final_grid - identity
            fstats = _norm_delta_bhwc_to_px_stats(final_delta)
            for k, v in fstats.items():
                diag[f"final_grid_disp_px_{k}"] = v

            residual_delta = final_grid - param_grid
            rstats = _norm_delta_bhwc_to_px_stats(residual_delta)
            for k, v in rstats.items():
                diag[f"residual_contrib_px_{k}"] = v

            denom = float(diag.get("param_grid_disp_px_mean", 0.0))
            if denom > 1e-12:
                diag["residual_to_param_disp_ratio_mean"] = float(rstats["mean"] / denom)
            else:
                diag["residual_to_param_disp_ratio_mean"] = 0.0

    return diag


def _extract_image_range_diagnostics(
    *,
    input_image: Tensor,
    target_image: Tensor,
    pred_image: Tensor,
) -> dict[str, float]:
    return {
        "input_min": _tensor_scalar(input_image.amin()),
        "input_max": _tensor_scalar(input_image.amax()),
        "target_min": _tensor_scalar(target_image.amin()),
        "target_max": _tensor_scalar(target_image.amax()),
        "pred_min": _tensor_scalar(pred_image.amin()),
        "pred_max": _tensor_scalar(pred_image.amax()),
    }


def _validate_model_outputs(raw_outputs: dict[str, Any], batch_size: int) -> None:
    if "params" not in raw_outputs:
        raise ValueError("model output must include key 'params'")

    params = raw_outputs["params"]
    if not torch.is_tensor(params):
        raise ValueError("model output 'params' must be a Tensor")
    if params.ndim != 2 or params.shape[0] != batch_size or params.shape[1] < 8:
        raise ValueError(
            f"model output 'params' must have shape [B,8+] with batch {batch_size}, got {tuple(params.shape)}"
        )


def _validate_residual_layout(residual: Tensor) -> None:
    if residual.ndim != 4:
        raise ValueError("residual_flow tensor must be 4D BCHW or BHWC")
    if residual.shape[1] == 2:
        return
    if residual.shape[-1] == 2:
        return
    raise ValueError(f"residual_flow expected BCHW/BHWC with 2 channels, got shape {tuple(residual.shape)}")


def forward_loss_step(
    *,
    model: nn.Module,
    batch: dict[str, Any],
    loss_fn: CompositeLoss,
    warp_backend: WarpBackend,
    stage: StageToggles,
    amp_enabled: bool,
    device: torch.device,
) -> TrainStepResult:
    b = _batch_to_device(batch, device)
    if "input_image" not in b or "target_image" not in b:
        raise ValueError("batch must include keys: input_image, target_image")

    input_image = b["input_image"]
    target_image = b["target_image"]

    if not torch.is_tensor(input_image) or not torch.is_tensor(target_image):
        raise ValueError("input_image and target_image must be tensors")
    if input_image.shape != target_image.shape:
        raise ValueError(
            f"input_image and target_image must have identical shape, got {tuple(input_image.shape)} vs {tuple(target_image.shape)}"
        )

    batch_size = int(input_image.shape[0])

    # Keep model forward under autocast for throughput, but run warp/loss in FP32 for numeric stability.
    # This avoids persistent GradScaler step skips caused by half-precision geometry/loss paths.
    with autocast_context(enabled=amp_enabled, device=device):
        raw_outputs = model(input_image)
    if not isinstance(raw_outputs, dict):
        raise ValueError("model forward must return dict-like outputs")

    _validate_model_outputs(raw_outputs, batch_size=batch_size)

    params = raw_outputs["params"]
    residual_lowres = raw_outputs.get("residual_flow_lowres")
    if residual_lowres is None:
        residual_lowres = raw_outputs.get("residual_flow")

    if torch.is_tensor(residual_lowres):
        _validate_residual_layout(residual_lowres)

    use_residual = stage.use_residual and torch.is_tensor(residual_lowres)
    residual_for_warp = residual_lowres if use_residual else None

    if amp_enabled:
        input_for_warp = input_image.float()
        target_for_loss = target_image.float()
        params_for_warp = params.float()
        residual_for_warp = residual_for_warp.float() if torch.is_tensor(residual_for_warp) else None
    else:
        input_for_warp = input_image
        target_for_loss = target_image
        params_for_warp = params

    warp_outputs = warp_backend.warp(input_for_warp, params_for_warp, residual_for_warp)
    if "pred_image" not in warp_outputs:
        raise ValueError("warp backend output must include key 'pred_image'")
    if "warp_stats" not in warp_outputs or not isinstance(warp_outputs["warp_stats"], dict):
        raise ValueError("warp backend output must include dict key 'warp_stats'")

    pred_image = warp_outputs["pred_image"]
    if not torch.is_tensor(pred_image):
        raise ValueError("warp backend 'pred_image' must be a Tensor")
    if pred_image.shape != target_image.shape:
        raise ValueError(
            f"pred_image and target_image must have identical shape, got {tuple(pred_image.shape)} vs {tuple(target_image.shape)}"
        )

    final_grid = warp_outputs.get("final_grid") if stage.use_jacobian_penalty else None
    if stage.use_jacobian_penalty and final_grid is None:
        raise ValueError("stage requires Jacobian penalty but warp backend did not return 'final_grid'")
    residual_for_loss = residual_for_warp if stage.use_flow_regularizers else None

    try:
        total_loss, components = loss_fn(
            pred_image,
            target_for_loss,
            residual_flow_lowres=residual_for_loss,
            final_grid_bhwc=final_grid,
        )
    except RuntimeError as exc:
        if "non-finite" in str(exc).lower() or "nan" in str(exc).lower() or "inf" in str(exc).lower():
            raise FloatingPointError(f"Non-finite loss computation detected: {exc}") from exc
        raise

    diagnostics = {}
    diagnostics.update(_extract_model_diagnostics(raw_outputs, model))
    diagnostics.update(_extract_warp_diagnostics(warp_outputs))
    diagnostics.update(
        _extract_image_range_diagnostics(
            input_image=input_for_warp,
            target_image=target_for_loss,
            pred_image=pred_image,
        )
    )

    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite total loss detected (NaN/Inf).")
    for name, value in components.items():
        if not torch.isfinite(value):
            raise FloatingPointError(f"Non-finite loss component detected: {name}")
        diagnostics[f"loss_finite_{name}"] = 1.0

    return TrainStepResult(
        total_loss=total_loss,
        components=components,
        pred_image=pred_image,
        model_outputs=raw_outputs,
        warp_outputs=warp_outputs,
        diagnostics=diagnostics,
    )


def run_train_step(
    *,
    model: nn.Module,
    batch: dict[str, Any],
    loss_fn: CompositeLoss,
    warp_backend: WarpBackend,
    stage: StageToggles,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    grad_clip_norm: float | None,
    device: torch.device,
    debug_instrumentation: bool = False,
) -> TrainStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    params_before: dict[str, Tensor] = {}
    if debug_instrumentation:
        params_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    forward_start = time.perf_counter()
    step_out = forward_loss_step(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        warp_backend=warp_backend,
        stage=stage,
        amp_enabled=amp_enabled,
        device=device,
    )

    if debug_instrumentation:
        _sync_if_cuda(device)
    forward_end = time.perf_counter()

    optim_step_skipped = False

    backward_start = time.perf_counter()
    if scaler is not None and amp_enabled:
        prev_scale = float(scaler.get_scale())
        scaler.scale(step_out.total_loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        if debug_instrumentation:
            _sync_if_cuda(device)
        backward_end = time.perf_counter()

        step_start = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        if debug_instrumentation:
            _sync_if_cuda(device)
        step_end = time.perf_counter()

        new_scale = float(scaler.get_scale())
        optim_step_skipped = new_scale < prev_scale
        step_out.diagnostics["amp_scale_before"] = prev_scale
        step_out.diagnostics["amp_scale_after"] = new_scale
        step_out.diagnostics["amp_scale_ratio"] = (new_scale / prev_scale) if prev_scale > 0.0 else 1.0
    else:
        step_out.total_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        if debug_instrumentation:
            _sync_if_cuda(device)
        backward_end = time.perf_counter()

        step_start = time.perf_counter()
        optimizer.step()
        if debug_instrumentation:
            _sync_if_cuda(device)
        step_end = time.perf_counter()

    if debug_instrumentation:
        step_out.diagnostics["timing_forward_ms"] = (forward_end - forward_start) * 1000.0
        step_out.diagnostics["timing_backward_ms"] = (backward_end - backward_start) * 1000.0
        step_out.diagnostics["timing_optim_step_ms"] = (step_end - step_start) * 1000.0
    step_out.diagnostics["optim_step_skipped"] = 1.0 if optim_step_skipped else 0.0

    if debug_instrumentation:
        step_out.diagnostics.update(_collect_group_grad_norms(model))
    if debug_instrumentation and params_before:
        step_out.diagnostics.update(_collect_group_param_deltas(model, params_before))
    elif debug_instrumentation:
        step_out.diagnostics.update(
            {
                "param_delta_backbone": 0.0,
                "param_delta_param_head": 0.0,
                "param_delta_residual_head": 0.0,
                "param_delta_other": 0.0,
            }
        )

    return step_out


def run_eval_step(
    *,
    model: nn.Module,
    batch: dict[str, Any],
    loss_fn: CompositeLoss,
    warp_backend: WarpBackend,
    stage: StageToggles,
    amp_enabled: bool,
    device: torch.device,
) -> TrainStepResult:
    model.eval()
    with torch.no_grad():
        return forward_loss_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            warp_backend=warp_backend,
            stage=stage,
            amp_enabled=amp_enabled,
            device=device,
        )
