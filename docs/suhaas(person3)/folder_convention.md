# Folder Convention (Person 3)

## Purpose
Define stable folder and filename conventions for data, proxy scoring, QA, and submission packaging.

## Expected Working Tree
```text
project-root/
  data/
    val_corrected/
    pred_val/
    test/
    pred_test_param/
    pred_test_hybrid/

  scripts/
    validate_proxy.py
    build_submission_zip.py
    run_ablation.py  # planned

  outputs/
    proxy/
      proxy_report.csv
      fail_report.csv
    selected/
    submissions/
      submission_vXX.zip
      manifest_vXX.json
    reports/
      experiment_log.csv

  docs/
    suhaas(person3)/
      metric_spec.md
      pipeline_spec.md
      folder_convention.md
      role_person_c.md
```

## Naming Rules
- Validation filenames must match between `data/val_corrected` and `data/pred_val`.
- Test prediction filenames must match test-set IDs in `data/test`.
- Prefer exact extension match across source and prediction folders.
- If extensions differ, mapping must be by stem and documented.

## Image Rules
- For a given ID, prediction and target must have identical width/height.
- Images must be RGB unless competition rules explicitly allow alternatives.
- Avoid alpha channel unless explicitly required.

## Safety Rules
- No black borders or empty corners in selected outputs.
- All outputs must decode successfully before zipping.
- Keep deterministic naming and manifest structure for reproducibility.