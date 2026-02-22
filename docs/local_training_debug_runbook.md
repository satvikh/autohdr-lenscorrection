# Local Training Debug Runbook

This runbook is for fast local diagnosis (no full training runs) of flat proxy behavior and residual-branch inactivity.

## Preconditions
- Use the project venv.
- Set `KMP_DUPLICATE_LIB_OK=TRUE`.
- On this Windows machine, use `AUTOHDR_NUM_WORKERS=0`.
- Keep dataset immutable; only write split CSVs and outputs.

## Debug Modes
1. Overfit Stage1 (`local_debug_overfit_stage1`):
- Goal: prove the model can overfit 32-64 pairs and beat identity on that same subset.
- Uses train==val split intentionally.

2. Stage2 residual probe (`local_debug_stage2_probe`):
- Goal: confirm residual branch is active and contributes non-zero displacement.
- Logs residual gradients, parameter deltas, residual displacement, and probe checksums.

3. Stage3 probe (`local_debug_stage3_probe`):
- Goal: verify fine-tune stage remains active and stable.

## New Diagnostics (enabled by `debug_instrumentation: true`)
- Gradient norms:
  - `grad_norm_backbone`
  - `grad_norm_param_head`
  - `grad_norm_residual_head`
- Parameter deltas:
  - `param_delta_backbone`
  - `param_delta_param_head`
  - `param_delta_residual_head`
- Residual health:
  - `residual_head_grad_is_zero`
  - `residual_head_param_delta_is_zero`
  - inactive residual warning when magnitude remains below threshold for configured patience
- Loss clarity:
  - raw: `flow_tv`, `flow_mag`, `jacobian`, etc.
  - weighted: `flow_tv_weighted`, `flow_mag_weighted`, `jacobian_weighted`, etc.
- Geometry/output magnitude:
  - `param_grid_disp_px_*`
  - `final_grid_disp_px_*`
  - `residual_contrib_px_*`
  - `residual_to_param_disp_ratio_mean`
- Validation probe:
  - `probe_pred_checksum`
  - `probe_final_grid_checksum`
  - `probe_params_checksum`
  - `probe_proxy_total_score`
  - `probe_pixel`, `probe_ssim`, `probe_edge`
- Performance (when `debug_perf_enabled: true`):
  - `data_time_ms`, `timing_forward_ms`, `timing_backward_ms`, `timing_optim_step_ms`
  - `batch_time_ms`, `samples_per_sec`
  - `cuda_mem_allocated_mb`, `cuda_mem_reserved_mb`, `max_cuda_mem_allocated_mb`

## Good vs Bad Signals
Good:
- Proxy and probe scores move at high precision across epochs.
- Probe checksums change across epochs.
- Stage2/3 residual norms and residual contribution are non-zero.
- Residual head gradient norm and param delta are non-zero.
- Overfit run beats identity on the exact same subset.

Bad:
- Probe checksums unchanged while parameter deltas are non-zero.
- Residual metrics stay at tiny thresholds with repeated inactivity warnings.
- Identity baseline outperforms trained model on same subset.
- Performance logs show high data-time and low throughput with idle GPU.

## Helper Scripts
- `scripts/debug_prepare_overfit_split.py`
  - Creates train/val identical CSVs for overfit checks.
- `scripts/debug_compare_identity.py`
  - Computes identity vs model proxy on the same subset and prints JSON summary.
