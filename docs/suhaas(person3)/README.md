# Person 3 Docs (Suhaas)

Person 3 owns Data + Proxy Scorer + QA + Submission Tooling.

## Ownership Scope
- Dataset audit and split generation.
- Data loading/transforms for train/val.
- Proxy metric components and aggregate scoring.
- QA checks and submission packaging.
- Experiment and submission manifest logging.

## Owned Code Paths
- `src/data/*`
- `src/metrics/*`
- `src/qa/*`
- `scripts/audit_dataset.py`
- `scripts/make_splits.py`
- `scripts/validate_proxy.py`
- `scripts/build_submission_zip.py`
- `tests/test_proxy_metrics.py`

## Core Documents
- `role_person_c.md`
- `person3_specdoc.md`
- `concurrency_plan_person3.md`
- `execution_checklist_person3.md`
- `pipeline_spec.md`
- `metric_spec.md`
- `folder_convention.md`

## Integration Dependencies
- Consumes model outputs from Person 2 for proxy validation.
- Consumes safety/fallback metadata from Person 1.
- Provides data contract consumed by Person 2 training.
