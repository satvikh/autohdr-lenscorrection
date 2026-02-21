# Person 1 Docs (Satvik)

Person 1 owns Geometry + Inference + Safety.

## Ownership Scope
- Geometry conventions and coordinate math.
- Parametric grid generation and residual fusion.
- Full-resolution inference pipeline.
- Safety checks and fallback hierarchy.

## Owned Code Paths
- `src/geometry/*`
- `src/inference/*`
- `tests/test_coords.py`
- `tests/test_parametric_warp.py`
- `tests/test_warp_identity.py`
- `tests/test_inference_pipeline.py`

## Core Documents
- `person1_specdoc.md`
- `concurrency_plan_person1.md`
- `execution_checklist_person1.md`
- `person3b_specdoc.md`
- `person3b_contract.md`
- `person3b_execution_checklist.md`

## Integration Dependencies
- Depends on model output contract from Person 2.
- Provides geometry API used by Person 2 training and Person 3 QA/safety reporting.
- Must keep inference metadata schema stable for Person 3 tooling.

## Rules
- Do not modify model/data/metrics/qa internals without explicit coordination.
- Contract changes must be reflected in `docs/contracts.md`.
