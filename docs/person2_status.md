# Person 2 Status (Model + Losses + Training)

## 1) Current State in Person 2 Scope
Implemented modules:

### `src/models/*`
- `src/models/coord_channels.py`
- `src/models/backbones.py`
- `src/models/heads_parametric.py`
- `src/models/heads_residual.py`
- `src/models/hybrid_model.py`
- `src/models/__init__.py`

### `src/losses/*`
- `src/losses/pixel.py`
- `src/losses/ssim_loss.py`
- `src/losses/gradients.py`
- `src/losses/flow_regularizers.py`
- `src/losses/jacobian_loss.py`
- `src/losses/composite.py`
- `src/losses/__init__.py`

### `src/train/*`
- `src/train/protocols.py`
- `src/train/warp_backends.py`
- `src/train/amp_utils.py`
- `src/train/optim.py`
- `src/train/checkpointing.py`
- `src/train/logging_utils.py`
- `src/train/stage_configs.py`
- `src/train/train_step.py`
- `src/train/engine.py`
- `src/train/__init__.py`

### `configs/*` in Person 2 ownership
- `configs/model/debug_smoke.yaml`
- `configs/model/resnet34_baseline.yaml`
- `configs/loss/debug_smoke.yaml`
- `configs/loss/stage1_param_only.yaml`
- `configs/loss/stage2_hybrid.yaml`
- `configs/loss/stage3_finetune.yaml`
- `configs/train/debug_smoke.yaml`
- `configs/train/stage1_param_only.yaml`
- `configs/train/stage2_hybrid.yaml`
- `configs/train/stage3_finetune.yaml`

### Person 2 tests
- `tests/test_model_shapes.py`
- `tests/test_loss_components.py`
- `tests/test_train_step_smoke.py`

## 2) Completed Functionality
- Coordinate channels (x, y, r) with explicit shape/range behavior.
- Backbone wrappers:
  - `resnet34`, `resnet50` via torchvision (when available)
  - `tiny` fallback backbone for smoke tests.
- Parametric head:
  - bounded outputs for global params
  - configurable bounds
  - near-identity safe initialization.
- Residual flow head:
  - FPN-like multi-scale decoder
  - low-res 2-channel flow output
  - `tanh * max_disp`
  - final conv zero-init.
- Hybrid model wrapper:
  - outputs `params`, `residual_flow`, `residual_flow_lowres`, `residual_flow_fullres`, `debug_stats`
  - optional `pred_image` via injected warp backend.
- Loss stack:
  - L1/Charbonnier
  - SSIM
  - Sobel edge magnitude
  - gradient orientation cosine
  - flow TV/magnitude/curvature regularizers
  - differentiable Jacobian foldover penalty
  - stage-aware `CompositeLoss` with component dictionary.
- Training stack:
  - dependency-injected warp backend protocol
  - mock backend and Person1 geometry backend adapter
  - modular train/eval step
  - optimizer/scheduler factories
  - AMP helpers
  - gradient clipping
  - checkpoint save/load
  - training engine with train/val loops and checkpointing.
- Stage toggles implemented:
  - `stage1_param_only`
  - `stage2_hybrid`
  - `stage3_finetune`.

## 3) Stubbed / Awaiting External Integration
- Person 3 data pipeline and dataloaders are not integrated yet; engine currently expects iterable batches with contract keys.
- Person 3 proxy scorer is not wired into validation loop yet; current validation reports loss components.
- Training entry scripts (`scripts/train_stage1.py`, `scripts/train_stage2.py`, `scripts/train_stage3.py`) are still pending in Person 2 lane.

## 4) Interfaces Expected from Person 1 (Warp Backend)
Primary geometry API (already available and used by adapter backend):
- `build_parametric_grid(params, height, width, align_corners, device, dtype) -> Tensor[B,H,W,2]`
- `upsample_residual_flow(flow_lr, target_h, target_w, align_corners) -> Tensor[B,H,W,2]`
- `fuse_grids(param_grid, residual_flow) -> Tensor[B,H,W,2]`
- `warp_image(image, grid, mode, padding_mode, align_corners) -> Tensor[B,C,H,W]`
- `jacobian_stats(grid) -> dict`

Expected behavior:
- BHWC grid convention, `(x, y)` ordering.
- `align_corners=True` globally.
- backward warp only.

Person 2 adapter implementation:
- `src/train/warp_backends.py::Person1GeometryWarpBackend`

## 5) Interfaces Expected from Person 3 (Proxy Scorer)
Expected entry point contract:
- `compute_proxy_score(pred, gt, config) -> dict`

Expected return schema:
- `total_score: float`
- `sub_scores: {edge, line, grad, ssim, mae}`
- `flags: {hard_fail: bool, reasons: list[str]}`

Planned integration point:
- validation hook inside `src/train/engine.py` (add optional proxy callback once Person 3 module is available).

## 6) Commands for Smoke Tests / Training
### Run Person 2 tests only
```bash
# Windows/Anaconda OpenMP workaround (if needed):
# set KMP_DUPLICATE_LIB_OK=TRUE
pytest -q tests/test_model_shapes.py tests/test_loss_components.py tests/test_train_step_smoke.py
```

### Run all tests
```bash
# Windows/Anaconda OpenMP workaround (if needed):
# set KMP_DUPLICATE_LIB_OK=TRUE
pytest -q
```

### Suggested smoke config usage (after stage scripts are added)
```bash
python scripts/train_stage1.py --model-config configs/model/debug_smoke.yaml --loss-config configs/loss/debug_smoke.yaml --train-config configs/train/debug_smoke.yaml
```

## 7) Immediate Next Steps in Person 2 Lane
1. Add stage training entry scripts under `scripts/train_stage*.py` to load YAML configs and run `TrainerEngine`.
2. Add optional validation callback interface for Person 3 proxy scorer.
3. Add metric/log export helpers for easier submission calibration with Person 3.
