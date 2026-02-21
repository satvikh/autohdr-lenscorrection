# Implementation Roadmap

This roadmap aligns directly with the final concurrency plan.

## Phase 0 (0-90 min): Alignment and Contract Freeze
Shared deliverables:
- Repo scaffold.
- `docs/contracts.md` and `docs/owners.md` finalized.
- Branches created and ownership assigned.

Done criteria:
- All interface keys/signatures frozen.
- Everyone can start independent implementation.

## Phase 1 (Hours 1-6): Parallel Foundations
### Person 1
- `src/geometry/coords.py`
- `src/geometry/parametric_warp.py`
- `src/geometry/warp_ops.py`
- geometry identity tests

### Person 2
- `src/models/backbones.py`
- `src/models/coord_channels.py`
- `src/models/heads_parametric.py`
- `src/models/heads_residual.py`
- `src/models/hybrid_model.py` (initial)
- `src/train/engine.py` (skeleton)
- `src/train/optim.py`

### Person 3
- `scripts/audit_dataset.py`
- `scripts/make_splits.py`
- `src/data/dataset.py`
- `src/data/transforms.py`
- `src/metrics/proxy_edge.py`
- `src/metrics/proxy_ssim_mae.py`
- `src/metrics/proxy_score.py` (stub)

Integration target:
- Phase 1 checkpoint passed.

## Phase 2 (Hours 6-12): First End-to-End Parametric Loop
### Person 1
- `src/geometry/residual_fusion.py`
- `src/geometry/jacobian.py`
- `src/geometry/sanity_tests.py`

### Person 2
- `src/losses/pixel.py`
- `src/losses/ssim_loss.py`
- `src/losses/gradients.py`
- `src/losses/composite.py` (stage 1)
- `scripts/train_stage1.py`

### Person 3
- `src/metrics/proxy_line.py`
- `src/metrics/proxy_gradient.py`
- `src/metrics/hardfail_checks.py`
- finalize `proxy_score.py`
- `scripts/validate_proxy.py`
- `src/qa/filename_check.py`
- `src/qa/image_integrity_check.py`

Integration target:
- First parametric-only train/val + proxy execution.

## Phase 3 (Hours 12-20): First Submission Path and Hybrid Prep
### Person 1
- `src/inference/predictor.py`
- `src/inference/writer.py`
- `src/inference/safety.py`
- `src/inference/fallback.py`
- `scripts/infer_test.py`

### Person 2
- `src/losses/flow_regularizers.py`
- `src/losses/jacobian_loss.py`
- stage-2 updates in `src/losses/composite.py`
- `scripts/train_stage2.py`

### Person 3
- `scripts/build_submission_zip.py`
- `src/qa/submission_manifest.py`
- baseline proxy calibration and reporting outputs

Integration target:
- First external baseline submission.

## Phase 4 (Hours 20-30): Hybrid Score Push
### Person 1
- hybrid full-res inference integration
- stronger safety thresholds
- full fallback hierarchy hardening

### Person 2
- Stage 2 hybrid training
- Stage 3 fine-tuning
- checkpoint ranking by proxy score

### Person 3
- `scripts/run_ablation.py`
- experiment tracking log
- comparative proxy reporting

Integration target:
- Hybrid checkpoint that beats parametric-only baseline.

## Phase 5 (Final): Optional TTO and Final Hardening
- Optional bounded TTO path if stable and beneficial.
- Final QA pass and submission packaging.
- Archive manifest + experiment log + checkpoint metadata.

## Exit Criteria
- Role-specific done criteria achieved for P1/P2/P3.
- Reproducible, QA-clean, contract-stable submission pipeline.