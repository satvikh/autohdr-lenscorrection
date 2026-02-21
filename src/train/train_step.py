from __future__ import annotations

from dataclasses import dataclass

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
    model_outputs: dict[str, Tensor]
    warp_outputs: dict[str, Tensor]


def _batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def forward_loss_step(
    *,
    model: nn.Module,
    batch: dict[str, Tensor],
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

    with autocast_context(enabled=amp_enabled, device=device):
        raw_outputs = model(input_image)
        if "params" not in raw_outputs:
            raise ValueError("model output must include key 'params'")

        params = raw_outputs["params"]
        residual_lowres = raw_outputs.get("residual_flow_lowres")
        if residual_lowres is None:
            residual_lowres = raw_outputs.get("residual_flow")

        use_residual = stage.use_residual and residual_lowres is not None
        residual_for_warp = residual_lowres if use_residual else None

        warp_outputs = warp_backend.warp(input_image, params, residual_for_warp)
        if "pred_image" not in warp_outputs:
            raise ValueError("warp backend output must include key 'pred_image'")

        pred_image = warp_outputs["pred_image"]
        final_grid = warp_outputs.get("final_grid") if stage.use_jacobian_penalty else None
        residual_for_loss = residual_for_warp if stage.use_flow_regularizers else None

        total_loss, components = loss_fn(
            pred_image,
            target_image,
            residual_flow_lowres=residual_for_loss,
            final_grid_bhwc=final_grid,
        )

    return TrainStepResult(
        total_loss=total_loss,
        components=components,
        pred_image=pred_image,
        model_outputs=raw_outputs,
        warp_outputs=warp_outputs,
    )


def run_train_step(
    *,
    model: nn.Module,
    batch: dict[str, Tensor],
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
    batch: dict[str, Tensor],
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