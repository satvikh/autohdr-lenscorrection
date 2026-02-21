# Person 1 Spec Doc: Geometry + Inference Lead

## 1. Purpose
This document defines exactly how Person 1 should execute the Geometry + Inference track for the Automatic Lens Correction project, with clear interfaces, sequencing, acceptance criteria, and failure controls.

Primary objective:
- Own all warp mathematics and full-resolution inference.
- Ensure outputs are geometrically correct, stable, safe, and submission-ready.

Out of scope (owned by others):
- Backbone/heads/loss design and core training loop (Person 2).
- Dataset audit, proxy metrics, QA packaging scripts (Person 3).

## 2. Success Criteria (Definition of Done)
Person 1 is complete when all are true:
1. Geometry tests pass (`coords`, identity warp, sanity checks, Jacobian checks).
2. Full-resolution inference works end-to-end on test images.
3. Safety checks and fallback hierarchy are active.
4. Inference outputs are valid JPEGs and stable across representative samples.
5. Geometry API is documented and consumed without contract mismatches.

## 3. Ownership and File Boundaries
Person 1 owns:
- `src/geometry/*`
- `src/inference/*`
- `tests/test_coords.py`
- `tests/test_parametric_warp.py`
- `tests/test_warp_identity.py`
- `tests/test_inference_pipeline.py`

Person 1 should avoid editing:
- `src/models/*`, `src/losses/*`, `src/train/*`, `src/data/*`, `src/metrics/*`, `src/qa/*`
- Shared files unless coordinated owner approves (`configs/*`, `docs/contracts.md`, `requirements.txt`)

## 4. Contract Requirements
Person 1 must enforce these interfaces early in `docs/contracts.md`.

### 4.1 Geometry API (mandatory)
```python
build_parametric_grid(params, height, width, align_corners, device, dtype) -> Tensor[B,H,W,2]
upsample_residual_flow(flow_lr, target_h, target_w, align_corners) -> Tensor[B,H,W,2]
fuse_grids(param_grid, residual_flow) -> Tensor[B,H,W,2]
warp_image(image, grid, mode, padding_mode, align_corners) -> Tensor[B,C,H,W]
jacobian_stats(grid) -> dict
```

### 4.2 Shape and convention contract
- Warp type: backward warp only.
- Grid convention: `grid_sample` normalized coordinates in `[-1, 1]`.
- `align_corners`: one value globally fixed (recommend `True`), never mixed.
- Sampling: single fused warp pass only (never double resample in production path).
- Default padding: `border`.

### 4.3 Model integration assumptions
Model output (from Person 2) should include:
- `params`: global parametric coefficients.
- `residual_flow`: low-res residual (or full-res if configured).

Person 1 should provide adapter handling for both:
- parametric-only mode (no residual input)
- hybrid mode (parametric + residual fusion)

## 5. Technical Specification

### 5.1 Coordinate system policy
- Define explicit converters:
  - pixel `(u,v)` <-> normalized image `(x,y)`
  - normalized image `(x,y)` <-> grid coords `(gx,gy)`
- Preserve deterministic formulas in one place (`coords.py`).
- Add round-trip tests with tight tolerance to prevent silent drift.

### 5.2 Parametric warp model
Implement Brown-Conrady-style bounded parameters:
- Radial: `k1, k2, k3`
- Tangential: `p1, p2`
- Center offsets: `dcx, dcy`
- Zoom/crop: `s`
- Optional: `aspect` (behind feature flag)

Use bounded outputs from model (already constrained upstream), but keep runtime clamps as safety guardrails.

Starter ranges (can be tuned by team):
- `k1 ∈ [-0.6, 0.6]`
- `k2 ∈ [-0.3, 0.3]`
- `k3 ∈ [-0.15, 0.15]`
- `p1,p2 ∈ [-0.03, 0.03]`
- `dcx,dcy ∈ [-0.08, 0.08]`
- `s ∈ [0.90, 1.20]`
- `aspect ∈ [0.97, 1.03]` (optional)

### 5.3 Residual flow handling
- Expect low-res residual grid from model.
- Upsample to target HxW with same `align_corners` policy.
- Bound displacement at runtime (`max_disp` config, initial 4-8 px at 512 training scale).
- Fused final grid:
  - `G_final = G_param + Delta_G_residual`

### 5.4 Jacobian and foldover safety
Implement `jacobian.py`:
- Compute local Jacobian determinant estimates for warp smoothness/foldover detection.
- Return stats:
  - `%negative_det`
  - `det_min`, `det_p01`, `det_mean`
  - optional high-gradient area fraction
- Expose this for both offline diagnostics and runtime safety gate.

### 5.5 One-pass warping rule
Always warp original image once with `G_final`.
Never do sequential warps in inference output path.

## 6. Inference System Specification

### 6.1 `predictor.py`
Responsibilities:
1. Load checkpoint and model in eval mode.
2. Preprocess input image to model size.
3. Run forward pass.
4. Build parametric grid.
5. If residual exists, upsample + fuse.
6. Compute safety stats.
7. Apply fallback if unsafe.
8. Warp once at full resolution.
9. Return image + metadata (`mode_used`, safety flags, scalar stats).

### 6.2 `writer.py`
- Save deterministic JPEGs for submission constraints.
- Preserve naming policy expected by QA/packaging.
- Validate write success and dimensions.

### 6.3 `safety.py`
Safety checks should include:
- Out-of-bounds sample ratio.
- Black/invalid border ratio.
- Jacobian foldover threshold.
- Residual magnitude threshold.
- Optional patch worst-case distortion proxy.

Return structured result:
```python
{
  "safe": bool,
  "reasons": [str],
  "metrics": {...}
}
```

### 6.4 `fallback.py`
Fallback hierarchy (required):
1. Hybrid (`params + residual`)
2. Param-only
3. Conservative param-only (clamped params and/or tighter zoom)

If mode 1 fails safety -> try 2.
If mode 2 fails -> try 3.
If mode 3 fails -> emit best-effort output with hard warning flag for QA review.

### 6.5 `scripts/infer_test.py`
- Batch inference over all test files.
- Save outputs and run-level metadata (checkpoint id, config hash, timestamp, mode counts, safety trigger counts).

## 7. Implementation Plan by Phase

### Phase 0 (0-90 min): Alignment
Deliverables:
- Finalized geometry API signatures in `docs/contracts.md`.
- Confirmed convention decisions (`align_corners`, backward warp, shape formats).
- Branch created (example: `codex/feat-geometry-core`).

Acceptance:
- Person 2 can call stubs without guessing signatures.

### Phase 1 (Hours 1-6): Geometry core foundation
Build:
- `src/geometry/coords.py`
- `src/geometry/parametric_warp.py`
- `src/geometry/warp_ops.py`
- Tests: `test_coords.py`, `test_parametric_warp.py`, `test_warp_identity.py`

Acceptance:
- Identity warp reconstructs input within tolerance.
- Batched dummy params produce sensible warps.
- No NaNs in generated grids.

### Phase 2 (Hours 6-12): Integration + Jacobian utilities
Build:
- `src/geometry/residual_fusion.py`
- `src/geometry/jacobian.py`
- `src/geometry/sanity_tests.py`

Acceptance:
- `build_parametric_grid` works batched with real model output.
- `fuse_grids` works with dummy residuals.
- Jacobian stats run without NaNs and detect synthetic foldovers.

### Phase 3 (Hours 12-20): Full-res inference + baseline submission path
Build:
- `src/inference/predictor.py`
- `src/inference/writer.py`
- `src/inference/safety.py`
- `src/inference/fallback.py`
- `scripts/infer_test.py`

Acceptance:
- Full test set can be processed.
- JPEG outputs valid and dimensionally correct.
- Safety logs generated.
- Param-only fallback proven to trigger and recover.

### Phase 4 (Hours 20-30): Hybrid inference hardening
Build:
- Integrate residual path in full-res mode.
- Calibrate safety thresholds from validation evidence.
- Strengthen fallback policy and logging.

Acceptance:
- Hybrid full-res inference stable.
- Failures route to safe fallback automatically.
- Run logs show low unsafe-output escape rate.

### Phase 5 (Final stretch, optional)
Build:
- Optional support for bounded TTO hook (if team chooses).

Acceptance:
- TTO path is reversible and only retained when proxy improves.

## 8. Test Plan

### 8.1 Unit tests
- Coordinate round-trip tests.
- Identity warp test (image unchanged under neutral params).
- Parametric monotonic sanity tests for radial effects.
- Residual fusion shape/range tests.
- Jacobian synthetic tests (known foldover vs non-foldover).

### 8.2 Integration tests
- Model output dict -> geometry API -> warped prediction pipeline.
- Inference pipeline with fallback forced by synthetic unsafe condition.
- End-to-end file IO test for few sample images.

### 8.3 Visual QA checks
On sampled validation images, inspect:
- Straight-line correction quality.
- Border behavior.
- Artifacts from residual overreach.
- Stability across scene types.

## 9. Operational Rules for Person 1
1. Keep PRs small and module-scoped.
2. Do not modify other owners’ directories without coordination.
3. Include test evidence in every PR.
4. Raise contract changes immediately; do not silently change signatures.
5. Use coding agents only within Person 1 files and with explicit file targets.

## 10. Risks and Mitigations

Risk: coordinate mismatch (`align_corners`, sign, normalization).
- Mitigation: centralize formulas in `coords.py` + round-trip tests + checkpoint freeze.

Risk: residual flow destabilizes geometry.
- Mitigation: strict displacement bounds, Jacobian monitoring, fallback to param-only.

Risk: border artifacts reduce edge score.
- Mitigation: zoom/crop handling via `s`, `padding_mode='border'`, explicit border checks.

Risk: merge conflicts in shared configs/contracts.
- Mitigation: assign shared-file owner, integration checkpoints, small PR cadence.

## 11. Deliverables Checklist
- `src/geometry/coords.py`
- `src/geometry/parametric_warp.py`
- `src/geometry/residual_fusion.py`
- `src/geometry/warp_ops.py`
- `src/geometry/jacobian.py`
- `src/geometry/sanity_tests.py`
- `src/inference/predictor.py`
- `src/inference/writer.py`
- `src/inference/safety.py`
- `src/inference/fallback.py`
- `scripts/infer_test.py`
- `tests/test_coords.py`
- `tests/test_parametric_warp.py`
- `tests/test_warp_identity.py`
- `tests/test_inference_pipeline.py`
- Updates to `docs/contracts.md` for geometry/inference APIs

## 12. Immediate Next Actions (Person 1)
1. Create and commit `docs/contracts.md` geometry section with exact signatures.
2. Implement `coords.py` and identity warp tests first.
3. Ship minimal `parametric_warp.py` + `warp_ops.py` and validate with checkerboard debug.
4. Provide Person 2 a stable minimal API before deeper optimization work.
5. Build inference + fallback path as soon as Stage 1 checkpoint is available.
