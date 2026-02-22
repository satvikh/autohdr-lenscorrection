# Cloud H100 Execution Runbook

This runbook is for executing full cloud training on RunPod H100 with the stabilized AMP/FP32 and diagnostics path.

## Repo + environment assumptions
- Repo: `/workspace/projects/autohdr-lenscorrection`
- Python: `/workspace/projects/autohdr-lenscorrection/.venv/bin/python`
- Data root: `/workspace/data/automatic-lens-correction/lens-correction-train-cleaned`
- Splits root (full): `/workspace/data/splits/full`
- Splits root (smoke): `/workspace/data/splits/smoke_2k`
- Logs: `/workspace/logs`

## Config set
- Models:
  - `configs/model/cloud_h100_stage1.yaml`
  - `configs/model/cloud_h100_stage2.yaml`
  - `configs/model/cloud_h100_stage3.yaml`
- Loss:
  - `configs/loss/cloud_h100_stage1.yaml`
  - `configs/loss/cloud_h100_stage2.yaml`
  - `configs/loss/cloud_h100_stage3.yaml`
- Train (smoke):
  - `configs/train/cloud_h100_stage1_smoke.yaml`
  - `configs/train/cloud_h100_stage2_smoke.yaml`
  - `configs/train/cloud_h100_stage3_smoke.yaml`
- Train (full):
  - `configs/train/cloud_h100_stage1.yaml`
  - `configs/train/cloud_h100_stage2.yaml`
  - `configs/train/cloud_h100_stage3.yaml`

## Required diagnostics to keep on
- `optim_step_skipped`
- `grad_nonfinite_count_backbone`
- `grad_nonfinite_count_param_head`
- `grad_nonfinite_count_residual_head`
- `amp_scale_before/after/ratio`
- probe checksums (`[val-probe]`)
- residual activity metrics (`residual_lowres_abs_*`, `residual_contrib_px_*`, ratio)
- perf/device metrics (`samples_per_sec`, `cuda_mem_*`, `data_time_ms`)
