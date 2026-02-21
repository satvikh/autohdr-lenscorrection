# AGENTS

Global operating instructions for human contributors and coding agents working in this repository.

## Mission
Deliver a leaderboard-safe automatic lens correction system with a geometry-first design, fast iteration loops, and reproducible submissions.

## Final Role Split
- Person 1 (Satvik): Geometry + Inference + Safety
- Person 2 (Kush): Model + Losses + Training
- Person 3 (Suhaas): Data + Proxy Scorer + QA + Submission Tooling

## Must-Read Order
1. `README.md`
2. `docs/DOC_INDEX.md`
3. `docs/contracts.md`
4. `docs/owners.md`
5. `docs/OWNERSHIP_AND_CONCURRENCY.md`
6. `docs/INTEGRATION_CHECKPOINTS.md`
7. Your person folder docs under `docs/*(personX)/`

## Rule Precedence
If instructions conflict:
1. `docs/contracts.md` (interfaces and conventions)
2. `docs/owners.md` (ownership boundaries)
3. person-specific spec docs
4. all other docs

## Global Technical Rules
- Warp direction: backward warp only.
- Geometry tensor layout in geometry modules: `BHWC`.
- Coordinate order: `(x, y)`.
- `align_corners=True` across all geometry and interpolation operations.
- One-pass fused sampling for final output generation.
- No sequential warps in production inference path.

## Coding Agent Rules
1. Scope agents to owned directories and explicit files only.
2. Do not ask agents to refactor the whole repo.
3. Require tests in every agent task.
4. Human review is required before commit.
5. Freeze interfaces before running large agent batches.

## Ownership Boundaries
### Person 1 Owns
- `src/geometry/*`
- `src/inference/*`
- `tests/test_coords.py`
- `tests/test_parametric_warp.py`
- `tests/test_warp_identity.py`
- `tests/test_inference_pipeline.py`

### Person 2 Owns
- `src/models/*`
- `src/losses/*`
- `src/train/*`
- `scripts/train_stage1.py`
- `scripts/train_stage2.py`
- `scripts/train_stage3.py`
- `tests/test_model_outputs.py`
- `tests/test_losses.py`

### Person 3 Owns
- `src/data/*`
- `src/metrics/*`
- `src/qa/*`
- `scripts/audit_dataset.py`
- `scripts/make_splits.py`
- `scripts/validate_proxy.py`
- `scripts/build_submission_zip.py`
- `tests/test_proxy_metrics.py`

### Shared High-Conflict Files
- `docs/contracts.md`
- `docs/owners.md`
- `configs/*` (planned)
- `requirements.txt`
- `README.md`

Only designated owners should directly modify these unless explicitly coordinated.

## PR Requirements
- PR scope aligned to ownership boundaries.
- Tests or validation evidence included.
- Interface changes declared explicitly.
- Cross-doc updates included when behavior changed.
- No repo-wide formatting unrelated to task scope.

## Integration Rules
- Prefer `dev` as integration branch; keep `main` stable.
- Use one integration captain for checkpoint merges.
- Freeze interfaces at each integration checkpoint.
- Resolve contract mismatches immediately when small.

## Project Definition of Done
- P1: Geometry tests + full-res inference + safety/fallback active.
- P2: Stage 1 and Stage 2 training stable, best checkpoints tracked.
- P3: Dataset audit/splits + reliable proxy + QA/submission tooling stable.
- Team: Reproducible, submission-ready pipeline with documented lineage.