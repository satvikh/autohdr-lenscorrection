# Person 3b Spec Doc: Proxy Scorer + QA + Submission Validation

## 1. Purpose
Define the implementation plan, interfaces, and acceptance criteria for Person 3b workstream:
- proxy scorer modules (`src/metrics/*`)
- validation runner (`scripts/validate_proxy.py`)
- submission QA (`src/qa/*`)
- submission packager (`scripts/build_submission_zip.py`)
- tests for metrics and QA checks

This document is the execution reference for Person 3b and should be used with:
- `docs/contracts.md`
- `docs/satvik(person1)/person3b_contract.md`

## 2. Scope and Ownership
Person 3b owns these paths only:
- `src/metrics/*`
- `src/qa/*`
- `scripts/validate_proxy.py`
- `scripts/build_submission_zip.py`
- `tests/test_proxy_metrics.py`
- `tests/test_qa_checks.py` (new)

Out of scope:
- `src/data/*` and split/audit scripts
- geometry/inference internals
- model/loss/training code

## 3. Objectives
1. Provide a stable proxy score that tracks correction quality with component visibility.
2. Prevent avoidable submission failures via deterministic QA gates.
3. Provide a deterministic packaging path for submission zips and manifests.

## 4. Deliverables
Required modules:
- `src/metrics/proxy_edge.py`
- `src/metrics/proxy_ssim_mae.py`
- `src/metrics/proxy_line.py`
- `src/metrics/proxy_gradient.py`
- `src/metrics/hardfail_checks.py`
- `src/metrics/proxy_score.py`

Required scripts:
- `scripts/validate_proxy.py`
- `scripts/build_submission_zip.py`

Required QA modules:
- `src/qa/filename_check.py`
- `src/qa/image_integrity_check.py`
- optional: `src/qa/submission_manifest.py`

Required tests:
- `tests/test_proxy_metrics.py`
- `tests/test_qa_checks.py`

## 5. Non-Negotiable Conventions
- Backward warp and geometry conventions are consumed, not redefined, by Person 3b.
- Do not sequentially warp in metrics/QA pipelines.
- Person 3b must treat dataset access as contract-based dict IO.
- All scoring and QA code must be deterministic for fixed inputs/config.

## 6. Dataset/Prediction Assumptions
Person 3b assumes dataset or loader yields dict keys:
```python
{
  "input_image": Tensor[C,H,W],
  "target_image": Tensor[C,H,W],
  "image_id": str,
  "orig_size": tuple[int, int],
  "metadata": dict,  # optional
}
```

Person 3b does not assume how loading/augmentation is implemented internally.

## 7. Scoring Design
Subscores (all normalized to `[0, 1]`, higher is better):
- edge similarity
- line consistency
- gradient orientation similarity
- SSIM
- MAE-derived pixel score (`1 - normalized_mae`)

Default aggregate weights:
- edge: `0.40`
- line: `0.22`
- gradient: `0.18`
- ssim: `0.15`
- mae: `0.05`

Hard-fail checks:
- catastrophic edge collapse
- extreme gradient mismatch
- invalid value/dtype/shape failures
- optional geometry-safety fail passthrough

Hard-fail policy:
- if `hard_fail == True`, total score is clamped by configured penalty rule.

## 8. Validation Runner Behavior
`scripts/validate_proxy.py` must:
1. load a val split manifest
2. map predictions to GT by `image_id`/filename
3. compute per-image proxy output via `compute_proxy_score(...)`
4. aggregate mean/median and component stats
5. emit machine-readable report JSON and optional CSV rows
6. return non-zero exit on contract/IO failures

## 9. Submission QA Behavior
`filename_check.py`:
- verify naming convention and one-to-one mapping vs expected ids
- detect missing, duplicate, extra files

`image_integrity_check.py`:
- verify decodability and non-empty images
- validate dimensions/channels match requirements
- validate no NaNs/inf after decode->array conversion path (if applicable)

Optional `submission_manifest.py`:
- generate manifest containing run metadata:
  - timestamp
  - commit hash
  - checkpoint id/path
  - config hash/path
  - file count
  - QA summary

## 10. Packaging Script Behavior
`scripts/build_submission_zip.py` must:
1. run filename + integrity checks
2. optionally generate manifest
3. create deterministic zip ordering by filename
4. verify zip contents after write
5. exit non-zero on any failed gate

## 11. Recommended Config Schema
Use a dict-like config with defaults:
```python
{
  "weights": {"edge": 0.40, "line": 0.22, "grad": 0.18, "ssim": 0.15, "mae": 0.05},
  "hardfail": {
    "enabled": True,
    "edge_min": 0.05,
    "grad_min": 0.05,
    "max_invalid_ratio": 0.0,
    "penalty_mode": "clamp",   # clamp | zero
    "penalty_value": 0.05
  },
  "io": {
    "strict_filename_match": True,
    "allowed_ext": [".jpg", ".jpeg", ".png"]
  }
}
```

## 12. Definition of Done (Person 3b)
1. All required modules and scripts exist and are wired.
2. `compute_proxy_score(...)` returns stable schema with total/subscores/flags.
3. Proxy validation script produces JSON summary on sample val set.
4. QA checks block malformed submission folders.
5. Zip packaging is deterministic across repeated runs.
6. `tests/test_proxy_metrics.py` and `tests/test_qa_checks.py` pass.

## 13. Risks and Mitigations
Risk: proxy diverges from leaderboard behavior.
- Mitigation: keep subscore visibility; calibrate thresholds/weights from submission feedback.

Risk: silent filename mismatch.
- Mitigation: strict one-to-one filename checks and fail-fast behavior.

Risk: non-deterministic packaging.
- Mitigation: fixed file sort + stable compression settings + deterministic manifest fields.
