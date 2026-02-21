# Pipeline Specification (Person 3)

## Inputs
- `data/val_corrected/`: ground truth corrected validation images.
- `data/pred_val/`: validation predictions for proxy scoring.
- `data/test/`: test images for filename baseline.
- `data/pred_test_param/`: test predictions from parametric model.
- `data/pred_test_hybrid/`: test predictions from hybrid model.

## Outputs
- `outputs/proxy/proxy_report.csv`: per-image and aggregate proxy scores.
- `outputs/proxy/fail_report.csv`: safety/hard-fail flags per image.
- `outputs/selected/`: final selected safe test outputs.
- `outputs/submissions/submission_vXX.zip`: submission artifact.
- `outputs/submissions/manifest_vXX.json`: metadata and lineage.
- `outputs/reports/experiment_log.csv`: experiment tracking ledger.

## Execution Order
1. Run proxy scoring on validation predictions.
2. Run QA checks on validation predictions.
3. If multiple test candidate folders exist, run selection policy.
4. Run QA checks on selected test outputs.
5. Build submission zip and manifest.
6. Append experiment/submission log row.

## Selection Policy
1. Prefer hybrid output when it passes safety checks.
2. Fallback to parametric output when hybrid fails.
3. If both fail, pick least risky output by border and worst-patch heuristics, and flag it.

## Key Constraints
- Preserve original resolution for every output image.
- Avoid black borders and catastrophic local artifacts.
- Ensure all files decode correctly.
- Ensure zip includes one valid image per required test ID.

## Integration Hooks
- Consumes `mode_used` and safety metadata from Person 1 inference.
- Uses model/checkpoint identifiers from Person 2 for manifest traceability.