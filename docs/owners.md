# Owners Map

Authoritative ownership map for code and documentation.

## Team Roles
- Person 1 (Satvik): Geometry + Inference + Safety
- Person 2 (Kush): Model + Losses + Training
- Person 3 (Suhaas): Data + Proxy Scorer + QA + Submission Tooling

## Directory Ownership
- `src/geometry/*`: Person 1
- `src/inference/*`: Person 1
- `src/models/*`: Person 2
- `src/losses/*`: Person 2
- `src/train/*`: Person 2
- `src/data/*`: Person 3
- `src/metrics/*`: Person 3
- `src/qa/*`: Person 3

## Script Ownership
- `scripts/infer_test.py`: Person 1
- `scripts/train_stage1.py`: Person 2
- `scripts/train_stage2.py`: Person 2
- `scripts/train_stage3.py`: Person 2
- `scripts/audit_dataset.py`: Person 3
- `scripts/make_splits.py`: Person 3
- `scripts/validate_proxy.py`: Person 3
- `scripts/build_submission_zip.py`: Person 3

## Test Ownership
- `tests/test_coords.py`: Person 1
- `tests/test_parametric_warp.py`: Person 1
- `tests/test_warp_identity.py`: Person 1
- `tests/test_inference_pipeline.py`: Person 1
- `tests/test_model_outputs.py`: Person 2
- `tests/test_losses.py`: Person 2
- `tests/test_proxy_metrics.py`: Person 3

## Shared Files and Owners
- `docs/contracts.md`: Primary owner Person 1, reviewers Person 2 + Person 3
- `docs/owners.md`: Primary owner Person 3, reviewers Person 1 + Person 2
- `configs/*` (planned): Primary owner Person 3, train-config subset review by Person 2
- `requirements.txt`: Primary owner Person 2, reviewers Person 1 + Person 3
- `README.md`: Primary owner Person 3, reviewers Person 1 + Person 2

## Integration Captain
- Default: Person 1
- Backup: Person 2

## Change Protocol
For shared files:
1. Announce planned change in team channel.
2. Open small PR with explicit interface or config impact.
3. Request review from both other owners.
4. Merge only after ack from integration captain.