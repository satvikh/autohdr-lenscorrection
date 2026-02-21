# Person 3b Execution Checklist

Use this before opening or merging a Person 3b PR.

## 1. Scope Guard
- [ ] Changes are limited to Person 3b-owned paths.
- [ ] No edits in `src/data/*` or split/audit scripts.
- [ ] Shared docs/contracts changes are explicitly called out.

## 2. Contract Compliance
- [ ] `compute_proxy_score(pred, gt, config)` exists and returns required schema.
- [ ] Subscore keys are exactly: `edge`, `line`, `grad`, `ssim`, `mae`.
- [ ] `flags` includes `hard_fail` and `reasons`.
- [ ] Validation and packaging CLIs match contract options.

## 3. Metric Sanity Gates
- [ ] GT-vs-GT average total score is near upper bound.
- [ ] Distorted/noisy baseline scores lower than GT-vs-GT.
- [ ] Hard-fail synthetic cases trigger expected penalties.
- [ ] No NaN/inf values in sub-scores or total score.

## 4. QA and Packaging Gates
- [ ] Missing, extra, and duplicate filenames are correctly reported.
- [ ] Broken/corrupt images are detected.
- [ ] Dimension mismatches are detected when size constraints are provided.
- [ ] Zip creation aborts on QA failure.
- [ ] Zip contents are deterministic (sorted, reproducible).

## 5. Test and Evidence
- [ ] `tests/test_proxy_metrics.py` passes.
- [ ] `tests/test_qa_checks.py` passes.
- [ ] Validation report JSON generated from a sample val run.
- [ ] PR includes sample command lines and report snippets.

## 6. Handoff Notes
- [ ] Any threshold or weight changes are documented.
- [ ] Any schema or CLI changes are reflected in `person3b_contract.md`.
- [ ] Dependencies for other owners are explicitly listed in PR description.

