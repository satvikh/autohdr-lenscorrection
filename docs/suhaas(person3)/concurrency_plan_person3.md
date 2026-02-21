# Person 3 Concurrency Plan (Data + Proxy + QA + Submission)

## Mission
Enable fast local iteration and protect submissions from avoidable failures.

## Phase 0 (0-90 min)
- Lock dataset and proxy API contracts in `docs/contracts.md`.
- Confirm shared config key ownership.

## Phase 1 (Hours 1-6)
Build:
- `scripts/audit_dataset.py`
- `scripts/make_splits.py`
- `src/data/dataset.py`
- `src/data/transforms.py`
- `src/metrics/proxy_edge.py`
- `src/metrics/proxy_ssim_mae.py`
- `src/metrics/proxy_score.py` (initial aggregate)
- `tests/test_proxy_metrics.py`

Done criteria:
- audit CSV and split files generated
- paired dataset loading works
- initial proxy outputs sensible values

## Phase 2 (Hours 6-12)
Build:
- `src/metrics/proxy_line.py`
- `src/metrics/proxy_gradient.py`
- `src/metrics/hardfail_checks.py`
- finalize `proxy_score.py`
- `scripts/validate_proxy.py`
- `src/qa/filename_check.py`
- `src/qa/image_integrity_check.py`

Done criteria:
- full proxy sub-score output available
- QA detects malformed outputs

## Phase 3 (Hours 12-20)
Build:
- `scripts/build_submission_zip.py`
- `src/qa/submission_manifest.py`
- baseline proxy calibration reports

Done criteria:
- automated zip + manifest generation
- submission QA automated
- baseline package ready for external upload

## Phase 4 (Hours 20-30)
Build:
- `scripts/run_ablation.py`
- experiment ranking logs
- variant comparison reports

Done criteria:
- runs are traceable and rankable quickly
- changes that improve proxy are identified reliably

## Phase 5 (Optional)
- TTO heuristics and final orchestration support

## Integration Checklist at Each Checkpoint
1. Validate model outputs can be scored without custom patches.
2. Validate safety metadata can be consumed by QA reports.
3. Validate submission outputs satisfy filename/dimension rules.
4. Validate experiment logs are complete and reproducible.