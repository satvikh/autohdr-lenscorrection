# Debugging Checklist

Use this checklist whenever scores or visuals regress.

## 1. Convention Sanity
- Confirm backward warp interpretation has not changed.
- Confirm geometry tensors are `BHWC` in geometry code.
- Confirm all geometry and interpolation paths use `align_corners=True`.
- Confirm coordinate order remains `(x, y)`.

## 2. Identity and Roundtrip Tests
- Run coordinate roundtrip tests.
- Run identity warp reconstruction test.
- Validate neutral params produce identity grid.

If any fail, stop and fix conventions before training.

## 3. Parametric Behavior Checks
- Sweep small positive and negative `k1` values.
- Verify radial behavior at corners is smooth and monotonic.
- Verify `dcx/dcy` move distortion center as expected.
- Verify `s` controls zoom/crop behavior predictably.

## 4. Grid and Sampling Checks
- Inspect min/max of grid values.
- Measure out-of-bounds ratio.
- Ensure grid has no NaNs/Infs.
- Confirm one-pass warp in inference output path.

## 5. Residual Path Checks (When Enabled)
- Validate input residual layout conversion (BCHW or BHWC).
- Validate pixel-to-normalized conversion math.
- Validate zero residual equals param-only path.
- Track residual magnitude and smoothness.

## 6. Safety and Fallback Checks
- Simulate known unsafe outputs and verify fallback triggers.
- Verify fallback order: hybrid -> param-only -> conservative param-only.
- Confirm metadata reports mode, reasons, and metrics.

## 7. Metric/Visual Mismatch Handling
If proxy score improves but visuals look worse:
- Check border artifact rates.
- Check local worst-patch errors.
- Check line-straightness and gradient-orientation components separately.
- Inspect per-image failures instead of only aggregate score.

## 8. Submission Stability Checks
- Decode all output files.
- Validate filename mapping.
- Confirm no accidental resolution changes.
- Save run manifest before packaging.