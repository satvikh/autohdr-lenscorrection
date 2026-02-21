# Person 3b Contract: Proxy + QA + Submission Interfaces

This contract freezes Person 3b interfaces so implementation can proceed in parallel without drift.

## 1. API Entry Point (Required)
`src/metrics/proxy_score.py` must expose:
```python
def compute_proxy_score(pred, gt, config):
    ...
```

Input contract:
- `pred`: corrected image array/tensor for one sample
- `gt`: ground-truth image array/tensor for same sample
- `config`: dict-like scoring config

Output contract (exact top-level keys):
```python
{
  "total_score": float,
  "sub_scores": {
    "edge": float,
    "line": float,
    "grad": float,
    "ssim": float,
    "mae": float
  },
  "flags": {
    "hard_fail": bool,
    "reasons": list[str]
  }
}
```

Notes:
- each sub-score must be bounded to `[0, 1]`
- `total_score` must be bounded to `[0, 1]` after hard-fail policy

## 2. Metric Module Contracts
`src/metrics/proxy_edge.py`
- public function:
```python
def edge_score(pred, gt, config) -> float:
    ...
```

`src/metrics/proxy_ssim_mae.py`
- public functions:
```python
def ssim_score(pred, gt, config) -> float:
    ...

def mae_score(pred, gt, config) -> float:
    ...
```

`src/metrics/proxy_line.py`
- public function:
```python
def line_score(pred, gt, config) -> float:
    ...
```

`src/metrics/proxy_gradient.py`
- public function:
```python
def gradient_score(pred, gt, config) -> float:
    ...
```

`src/metrics/hardfail_checks.py`
- public function:
```python
def evaluate_hard_fail(sub_scores, pred, gt, config) -> dict:
    # returns {"hard_fail": bool, "reasons": list[str]}
```

## 3. Aggregation Contract
`proxy_score.py` aggregation rule:
```python
weighted = (
  w_edge * edge +
  w_line * line +
  w_grad * grad +
  w_ssim * ssim +
  w_mae  * mae
)
```

Weight requirements:
- defaults: edge `0.40`, line `0.22`, grad `0.18`, ssim `0.15`, mae `0.05`
- if custom weights are provided, they must sum to `1.0` within tolerance

Hard-fail rules:
- if `hard_fail` is true and `penalty_mode == "clamp"`:
  - `total_score = min(weighted, penalty_value)`
- if `hard_fail` is true and `penalty_mode == "zero"`:
  - `total_score = 0.0`

## 4. Validation Script Contract
`scripts/validate_proxy.py` CLI (minimum):
```bash
python scripts/validate_proxy.py \
  --pred_dir <path> \
  --gt_dir <path> \
  --split_file <path> \
  --out_json <path>
```

Optional flags:
- `--config <path>`
- `--out_csv <path>`
- `--strict` (fail on any missing mapping)

Exit codes:
- `0`: completed and report written
- non-zero: contract violation, IO failure, or data mismatch

Required output JSON fields:
```json
{
  "n_expected": 0,
  "n_scored": 0,
  "n_missing_pred": 0,
  "n_extra_pred": 0,
  "mean_total_score": 0.0,
  "mean_sub_scores": {"edge": 0.0, "line": 0.0, "grad": 0.0, "ssim": 0.0, "mae": 0.0},
  "hard_fail_count": 0,
  "hard_fail_rate": 0.0
}
```

## 5. QA Module Contracts
`src/qa/filename_check.py`
- public function:
```python
def validate_filenames(pred_dir, expected_ids, allowed_ext=None) -> dict:
    # {
    #   "ok": bool,
    #   "missing": list[str],
    #   "extra": list[str],
    #   "duplicates": list[str]
    # }
```

`src/qa/image_integrity_check.py`
- public function:
```python
def validate_images(pred_dir, expected_size=None) -> dict:
    # {
    #   "ok": bool,
    #   "checked": int,
    #   "decode_failures": list[str],
    #   "size_mismatches": list[str],
    #   "mode_mismatches": list[str]
    # }
```

Optional `src/qa/submission_manifest.py`
- public function:
```python
def build_manifest(metadata: dict, qa_results: dict) -> dict:
    ...
```

## 6. Submission Packaging Contract
`scripts/build_submission_zip.py` CLI (minimum):
```bash
python scripts/build_submission_zip.py \
  --pred_dir <path> \
  --expected_ids <path> \
  --out_zip <path>
```

Optional flags:
- `--manifest_out <path>`
- `--checkpoint <id_or_path>`
- `--config <path>`
- `--strict`

Behavior:
- MUST run filename and integrity checks before zip creation
- MUST fail-fast on QA failure
- MUST include deterministic archive ordering

## 7. Cross-Team Boundary Contract
Person 3b assumptions from data layer:
- receives dict with required keys
- no dependency on data-loading internals

Person 3b assumptions from geometry/inference:
- may consume safety metadata if present
- does not modify geometry conventions

Person 3b does not edit:
- `src/data/*`
- `scripts/audit_dataset.py`
- `scripts/make_splits.py`

## 8. Test Contract
`tests/test_proxy_metrics.py` minimum required coverage:
- perfect match: score near `1.0`, no hard fail
- noisy/degraded match: score lower than perfect baseline
- hard-fail synthetic case: `hard_fail=True` and total score penalty applied
- schema stability: exact expected keys

`tests/test_qa_checks.py` minimum required coverage:
- filename checker catches missing/extra/duplicate
- integrity checker catches decode/size mismatch
- packaging script refuses zip when QA fails
- deterministic zip ordering on fixed inputs

