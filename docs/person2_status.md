# Person 2 Final Status (Model + Losses + Training)

## 1) Module Map (Person 2 Scope)

### Models (`src/models/*`)
- `coord_channels.py`: appends x/y/r coordinate channels for arbitrary `H/W`.
- `backbones.py`: `tiny`, `resnet34`, `resnet50` feature backbones.
- `heads_parametric.py`: bounded global parameter head with near-identity init.
- `heads_residual.py`: residual low-res flow head (`tanh * max_disp`, zero-init final conv).
- `hybrid_model.py`: unified model forward returning params + residual outputs + debug stats.

### Losses (`src/losses/*`)
- `pixel.py`: L1 / Charbonnier.
- `ssim_loss.py`: SSIM loss with finite guards.
- `gradients.py`: Sobel edge + gradient orientation losses with multiscale helper.
- `flow_regularizers.py`: TV / magnitude / curvature flow regularizers.
- `jacobian_loss.py`: Jacobian foldover penalty hook.
- `composite.py`: stage-aware weighted composite loss and component dictionary.

### Training (`src/train/*`)
- `protocols.py`: `WarpBackend` and `ProxyScorer` contracts.
- `warp_backends.py`: `Person1GeometryWarpBackend` + `MockWarpBackend`.
- `train_step.py`: train/eval step, stage gating, loss/warp contract checks, diagnostics.
- `engine.py`: loops, AMP/scaler, scheduler stepping, guardrails, checkpointing, logging.
- `optim.py`: AdamW + `cosine` / `onecycle` / `none` scheduler factory.
- `checkpointing.py`: save/load model + optimizer + scheduler + scaler state.
- `config_loader.py`: typed YAML parsing + schema validation.
- `proxy_hooks.py`: optional proxy scorer integration with resilient fallback.
- `debug_dump.py`: optional bounded image debug dump.

### Stage Scripts (`scripts/*`)
- `train_stage1.py`: stage1 train/validate entrypoint.
- `train_stage2.py`: stage2 entrypoint (default `--init-from outputs/runs/stage1_param_only/best.pt`).
- `train_stage3.py`: stage3 entrypoint (default `--init-from outputs/runs/stage2_hybrid/best.pt`).
- `_train_common.py`: shared CLI, fail-fast checks, loader wiring, resume/init/validate-only.

### Configs
- Model: `configs/model/debug_smoke.yaml`, `configs/model/resnet34_baseline.yaml`
- Loss: `configs/loss/debug_smoke.yaml`, `configs/loss/stage1_param_only.yaml`, `configs/loss/stage2_hybrid.yaml`, `configs/loss/stage3_finetune.yaml`
- Train: `configs/train/debug_smoke.yaml`, `configs/train/stage1_param_only.yaml`, `configs/train/stage2_hybrid.yaml`, `configs/train/stage3_finetune.yaml`

### Tests (Person 2-focused)
- `tests/test_model_shapes.py`
- `tests/test_loss_components.py`
- `tests/test_train_step_smoke.py`
- `tests/test_train_config_parsing.py`
- `tests/test_train_integration_hooks.py`

## 2) Integration Status

### Person 1 (geometry/inference): **Confirmed**
Integrated through `src/train/warp_backends.py` using:
- `build_parametric_grid`
- `upsample_residual_flow`
- `fuse_grids`
- `warp_image`
- `jacobian_stats`
- `evaluate_safety` (diagnostic report)

Train-time warp contract now enforced in Person 2 step:
- required: `pred_image`, `warp_stats`
- required for stage2/3 Jacobian penalty: `final_grid`
- optional: `param_grid`, `residual_flow_fullres_norm`

Jacobian semantics used in diagnostics:
- `negative_det_pct` higher is worse (foldovers).
- `det_min` / `det_p01` lower is worse.
- `oob_ratio` higher is worse.

### Person 3 (data/proxy): **Pending in current repo state**
- `src/data/*` not present.
- `src/metrics/*` not present.
- `src/qa/*` not present.

Hooks are ready and optional:
- External loaders via `--loader-module` + `--loader-fn`.
- Proxy scorer via train config (`proxy_enabled`, `proxy_module_path`, `proxy_function_name`).
- If scorer import fails, training remains non-fatal and logs warning.

## 3) Guardrails and Reliability
- Non-finite loss fail-fast with clear error (`fail_on_nonfinite_loss`).
- Parameter saturation warning with per-parameter names and fractions.
- Residual magnitude warning with actual observed values.
- CPU fallback if requested CUDA/MPS unavailable.
- Interrupt checkpoint: `interrupted.pt`.
- Config validation for stage/warp/scheduler names and numeric ranges.
- Stage/loss alignment checks in `_train_common.py`.

## 4) Run Commands (Copy-Paste)

### A) Synthetic smoke train
```bash
python scripts/train_stage1.py \
  --model-config configs/model/debug_smoke.yaml \
  --loss-config configs/loss/debug_smoke.yaml \
  --train-config configs/train/debug_smoke.yaml \
  --use-synthetic
```

### B) Stage1 real run
```bash
python scripts/train_stage1.py \
  --model-config configs/model/resnet34_baseline.yaml \
  --loss-config configs/loss/stage1_param_only.yaml \
  --train-config configs/train/stage1_param_only.yaml \
  --loader-module <person3_data_module> \
  --loader-fn build_train_val_loaders
```

### C) Stage2 real run
```bash
python scripts/train_stage2.py \
  --model-config configs/model/resnet34_baseline.yaml \
  --loss-config configs/loss/stage2_hybrid.yaml \
  --train-config configs/train/stage2_hybrid.yaml \
  --loader-module <person3_data_module> \
  --loader-fn build_train_val_loaders
```

### D) Stage3 real run
```bash
python scripts/train_stage3.py \
  --model-config configs/model/resnet34_baseline.yaml \
  --loss-config configs/loss/stage3_finetune.yaml \
  --train-config configs/train/stage3_finetune.yaml \
  --loader-module <person3_data_module> \
  --loader-fn build_train_val_loaders
```

### E) Resume example
```bash
python scripts/train_stage2.py \
  --resume-from outputs/runs/stage2_hybrid/last.pt \
  --loader-module <person3_data_module> \
  --loader-fn build_train_val_loaders
```

### F) Validate-only example
```bash
python scripts/train_stage2.py \
  --validate-only \
  --resume-from outputs/runs/stage2_hybrid/best.pt \
  --loader-module <person3_data_module> \
  --loader-fn build_train_val_loaders
```

## 5) Windows OpenMP Note
- In some Windows/Anaconda setups, you may hit duplicate OpenMP runtime errors.
- Workaround for local runs:
  - PowerShell: `$env:KMP_DUPLICATE_LIB_OK='TRUE'`
- Do not hardcode this in source; treat as environment-only workaround.

## 6) Top Priority Tuning Knobs
1. Per-stage `optimizer.lr` (largest stability/quality lever).
2. `grad_clip_norm` (reduce if unstable; raise if over-constrained).
3. Stage2/3 `jacobian_weight`, `flow_tv_weight`, `flow_mag_weight`.
4. `residual_max_disp` (model config) vs residual regularizer strengths.
5. Stage3 image-term balance: `edge_weight` / `grad_orient_weight` / `pixel_weight`.

## 7) Known Risks and Quick Detection
- **Risk:** parameter saturation at bounds.
  - Detect: `param_sat_frac_*` and `param_sat_frac_max` logs.
- **Risk:** unstable residual branch.
  - Detect: residual magnitude warnings + `residual_*` diagnostics.
- **Risk:** geometry foldovers.
  - Detect: `warp_negative_det_pct`, `warp_det_min`, `warp_oob_ratio`, `warp_safety_safe`.
- **Risk:** missing Person 3 runtime modules.
  - Detect: loader import errors / unresolved proxy warning during startup.
