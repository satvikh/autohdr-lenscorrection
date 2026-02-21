# Person 1 Concurrency Plan (Geometry + Inference + Safety)

## Mission
Deliver mathematically correct warp behavior and safe, full-resolution inference outputs.

## Phase 0 (0-90 min)
- Finalize geometry API in `docs/contracts.md`.
- Freeze coordinate conventions and parameter ordering.
- Create/confirm feature branches.

## Phase 1 (Hours 1-6)
Build:
- `src/geometry/coords.py`
- `src/geometry/parametric_warp.py`
- `src/geometry/warp_ops.py`
- geometry identity/roundtrip tests

Done criteria:
- identity warp reproduces input
- no NaNs/Infs in grid generation
- contract-aligned signatures

## Phase 2 (Hours 6-12)
Build:
- `src/geometry/residual_fusion.py`
- `src/geometry/jacobian.py`
- `src/geometry/sanity_tests.py`

Done criteria:
- batched parametric and hybrid grid support
- Jacobian stats compute reliably
- sanity tests pass

## Phase 3 (Hours 12-20)
Build:
- `src/inference/predictor.py`
- `src/inference/writer.py`
- `src/inference/safety.py`
- `src/inference/fallback.py`
- `scripts/infer_test.py`

Done criteria:
- full-res inference on test set
- safety stats logged
- fallback route operational

## Phase 4 (Hours 20-30)
Build:
- hybrid residual integration in inference
- threshold calibration
- stronger fallback hierarchy

Done criteria:
- hybrid inference stable
- unsafe outputs routed to safe fallback

## Phase 5 (Optional)
- bounded TTO hook support if stable and time permits

## Integration Checklist at Each Checkpoint
1. Verify model output keys from Person 2.
2. Verify metadata schema consumed by Person 3.
3. Verify no contract drift in geometry API.
4. Provide visual examples for edge/border sanity.