# Cloud H100 Execution Runbook

RunPod H100 (80GB) execution guide for smoke -> full training -> identity checks -> inference -> submission zip.

## Environment assumptions
- Repo: `/workspace/projects/autohdr-lenscorrection`
- Data root: `/workspace/data/automatic-lens-correction`
- Train pairs root: `/workspace/data/automatic-lens-correction/lens-correction-train-cleaned`
- Test images root: `/workspace/data/automatic-lens-correction/test-originals`
- Full splits root: `/workspace/data/splits/full_native`
- Smoke splits root: `/workspace/data/splits/smoke_2k`
- Logs dir: `/workspace/logs`
- Python venv active at `.venv`

## Dependency sanity
Install with:
`pip install -r requirements.txt`

`requirements.txt` already includes both `PyYAML` and `torchvision`.

## Config set in this runbook
- Model: `configs/model/cloud_h100_stage1.yaml`, `configs/model/cloud_h100_stage2.yaml`, `configs/model/cloud_h100_stage3.yaml`
- Loss: `configs/loss/cloud_h100_stage1.yaml`, `configs/loss/cloud_h100_stage2.yaml`, `configs/loss/cloud_h100_stage3.yaml`
- Train smoke: `configs/train/cloud_h100_stage1_smoke.yaml`, `configs/train/cloud_h100_stage2_smoke.yaml`, `configs/train/cloud_h100_stage3_smoke.yaml`
- Train full: `configs/train/cloud_h100_stage1.yaml`, `configs/train/cloud_h100_stage2.yaml`, `configs/train/cloud_h100_stage3.yaml`

## Metric alignment policy used in cloud train configs
- Proxy scoring enabled with explicit metric weights:
  - edge `0.40`, line `0.22`, grad `0.18`, ssim `0.15`, mae `0.05`
- Full-res proxy validation slice enabled:
  - smoke: `proxy_fullres_max_images=32`
  - full: `proxy_fullres_max_images=128`
- Checkpoint selection metric:
  - `best_metric_name: proxy_fullres_slice_total_score`
  - `best_metric_mode: max`
- Hard-fail policy for training selection:
  - `score_zero` (strict official behavior) for edge/regional hard-fail conditions.

## Native-resolution note
Real training runs at native resolution because `AUTOHDR_RESIZE_HW` is intentionally unset.

`synthetic_height: 256` and `synthetic_width: 256` in train YAMLs are synthetic diagnostics only and do not resize real data.

## H100 performance tuning
- Recommended full-run defaults:
  - `AUTOHDR_BATCH_SIZE=2`
  - `AUTOHDR_NUM_WORKERS=8`
  - `AUTOHDR_PIN_MEMORY=1`
- If OOM:
  - reduce `AUTOHDR_BATCH_SIZE` to `1`
  - optionally reduce workers to `4`
  - if needed, lower `proxy_fullres_max_images` in train config
- `max_steps_per_epoch` unset means full epoch over the selected split.

## Exact one-line commands
Run these from `/workspace/projects/autohdr-lenscorrection`.

### Base prep
`cd /workspace/projects/autohdr-lenscorrection && source .venv/bin/activate && mkdir -p /workspace/logs /workspace/outputs`

### Smoke: Stage 1/2/3
`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/smoke_2k AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/train_stage1.py --model-config configs/model/cloud_h100_stage1.yaml --loss-config configs/loss/cloud_h100_stage1.yaml --train-config configs/train/cloud_h100_stage1_smoke.yaml --loader-module src.data.real_loader --loader-fn build_train_val_loaders 2>&1 | tee /workspace/logs/cloud_h100_stage1_smoke.log`

`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/smoke_2k AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/train_stage2.py --model-config configs/model/cloud_h100_stage2.yaml --loss-config configs/loss/cloud_h100_stage2.yaml --train-config configs/train/cloud_h100_stage2_smoke.yaml --loader-module src.data.real_loader --loader-fn build_train_val_loaders --init-from outputs/runs/cloud_h100_stage1_smoke/best.pt 2>&1 | tee /workspace/logs/cloud_h100_stage2_smoke.log`

`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/smoke_2k AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/train_stage3.py --model-config configs/model/cloud_h100_stage3.yaml --loss-config configs/loss/cloud_h100_stage3.yaml --train-config configs/train/cloud_h100_stage3_smoke.yaml --loader-module src.data.real_loader --loader-fn build_train_val_loaders --init-from outputs/runs/cloud_h100_stage2_smoke/best.pt 2>&1 | tee /workspace/logs/cloud_h100_stage3_smoke.log`

### Full: Stage 1/2/3
`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/full_native AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/train_stage1.py --model-config configs/model/cloud_h100_stage1.yaml --loss-config configs/loss/cloud_h100_stage1.yaml --train-config configs/train/cloud_h100_stage1.yaml --loader-module src.data.real_loader --loader-fn build_train_val_loaders 2>&1 | tee /workspace/logs/cloud_h100_stage1.log`

`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/full_native AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/train_stage2.py --model-config configs/model/cloud_h100_stage2.yaml --loss-config configs/loss/cloud_h100_stage2.yaml --train-config configs/train/cloud_h100_stage2.yaml --loader-module src.data.real_loader --loader-fn build_train_val_loaders --init-from outputs/runs/cloud_h100_stage1/best.pt 2>&1 | tee /workspace/logs/cloud_h100_stage2.log`

`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/full_native AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/train_stage3.py --model-config configs/model/cloud_h100_stage3.yaml --loss-config configs/loss/cloud_h100_stage3.yaml --train-config configs/train/cloud_h100_stage3.yaml --loader-module src.data.real_loader --loader-fn build_train_val_loaders --init-from outputs/runs/cloud_h100_stage2/best.pt 2>&1 | tee /workspace/logs/cloud_h100_stage3.log`

### Identity comparisons (val subset)
`.venv/bin/python -c "import yaml,json; cfg=yaml.safe_load(open('configs/train/cloud_h100_stage3.yaml')); json.dump(cfg['proxy_config'], open('/workspace/logs/cloud_proxy_config.json','w'))"`

`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/full_native AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/debug_compare_identity.py --model-config configs/model/cloud_h100_stage1.yaml --checkpoint outputs/runs/cloud_h100_stage1/best.pt --stage stage1_param_only --subset val --max-batches 16 --device cuda --loader-module src.data.real_loader --loader-fn build_train_val_loaders --proxy-config-json /workspace/logs/cloud_proxy_config.json --json-out /workspace/logs/cloud_h100_identity_stage1.json 2>&1 | tee /workspace/logs/cloud_h100_identity_stage1.log`

`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/full_native AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/debug_compare_identity.py --model-config configs/model/cloud_h100_stage2.yaml --checkpoint outputs/runs/cloud_h100_stage2/best.pt --stage stage2_hybrid --subset val --max-batches 16 --device cuda --loader-module src.data.real_loader --loader-fn build_train_val_loaders --proxy-config-json /workspace/logs/cloud_proxy_config.json --json-out /workspace/logs/cloud_h100_identity_stage2.json 2>&1 | tee /workspace/logs/cloud_h100_identity_stage2.log`

`AUTOHDR_DATA_ROOT=/workspace/data/automatic-lens-correction/lens-correction-train-cleaned AUTOHDR_SPLIT_DIR=/workspace/data/splits/full_native AUTOHDR_REBUILD_SPLITS=0 AUTOHDR_BATCH_SIZE=2 AUTOHDR_NUM_WORKERS=8 AUTOHDR_PIN_MEMORY=1 AUTOHDR_RESIZE_HW= .venv/bin/python scripts/debug_compare_identity.py --model-config configs/model/cloud_h100_stage3.yaml --checkpoint outputs/runs/cloud_h100_stage3/best.pt --stage stage3_finetune --subset val --max-batches 16 --device cuda --loader-module src.data.real_loader --loader-fn build_train_val_loaders --proxy-config-json /workspace/logs/cloud_proxy_config.json --json-out /workspace/logs/cloud_h100_identity_stage3.json 2>&1 | tee /workspace/logs/cloud_h100_identity_stage3.log`

### Inference on 1000 test images
`.venv/bin/python scripts/infer_test.py /workspace/data/automatic-lens-correction/test-originals /workspace/outputs/cloud_h100_stage3_test_preds --checkpoint-id cloud_h100_stage3_best --checkpoint-path outputs/runs/cloud_h100_stage3/best.pt --model-config configs/model/cloud_h100_stage3.yaml --device cuda --config-path configs/train/cloud_h100_stage3.yaml 2>&1 | tee /workspace/logs/cloud_h100_infer_test.log`

### Build submission zip
`find /workspace/data/automatic-lens-correction/test-originals -maxdepth 1 -type f \\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \\) -printf '%f\n' | sed 's/\.[^.]*$//' | sort -u > /workspace/data/splits/full_native/test_ids.txt`

`.venv/bin/python scripts/build_submission_zip.py --pred_dir /workspace/outputs/cloud_h100_stage3_test_preds --ids_file /workspace/data/splits/full_native/test_ids.txt --out_zip /workspace/outputs/cloud_h100_stage3_submission.zip --strict 2>&1 | tee /workspace/logs/cloud_h100_submission_zip.log`

### Log summary checks
`grep -E "proxy_fullres_slice_total_score|proxy_hard_fail|optim_step_skipped|grad_nonfinite_count_|samples_per_sec|warp_safety_safe" /workspace/logs/cloud_h100_stage1.log /workspace/logs/cloud_h100_stage2.log /workspace/logs/cloud_h100_stage3.log | tail -n 200`

`grep -E "mode=|initial_safe=|final_safe=|Summary|unsafe_triggers|fallback_counts" /workspace/logs/cloud_h100_infer_test.log | tail -n 200`
