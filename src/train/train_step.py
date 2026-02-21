from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer

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
    if torch.is_tensor(params):
        diag["params_abs_mean"] = _tensor_scalar(params.abs().mean())
        diag["params_abs_max"] = _tensor_scalar(params.abs().amax())
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

    residual = raw_outputs.get("residual_flow_lowres")
    if residual is None:
        residual = raw_outputs.get("residual_flow")
    if torch.is_tensor(residual):
        diag["residual_lowres_abs_mean_px"] = _tensor_scalar(residual.abs().mean())
        diag["residual_lowres_abs_max_px"] = _tensor_scalar(residual.abs().amax())

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

    return diag


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

        warp_outputs = warp_backend.warp(input_image, params, residual_for_warp)
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
                target_image,
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

    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite total loss detected (NaN/Inf).")
    for name, value in components.items():
        if not torch.isfinite(value):
            raise FloatingPointError(f"Non-finite loss component detected: {name}")

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
) -> TrainStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step_out = forward_loss_step(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        warp_backend=warp_backend,
        stage=stage,
        amp_enabled=amp_enabled,
        device=device,
    )

    if scaler is not None and amp_enabled:
        scaler.scale(step_out.total_loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
    else:
        step_out.total_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

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
