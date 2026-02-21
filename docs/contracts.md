# Contracts: Geometry + Inference API (Phase 0)

## Scope
This document defines only the Person 1 contracts required in Phase 0:
- Geometry API
- Inference API

No training/model internals are defined here beyond integration inputs/outputs.

## Global Geometry Policies
- Warp direction: backward warp only.
- Sampling coordinate convention: PyTorch `grid_sample` normalized coordinates in `[-1, 1]`.
- `align_corners`: single global value for all geometry operations; set to `True` for this project and never mix values across modules.
- Warping rule: single-pass warp only in production inference (`G_final` must be fused before sampling).
- Default padding policy: `padding_mode="border"`.

## Geometry API (Mandatory, Exact Signatures)
```python
build_parametric_grid(params, height, width, align_corners, device, dtype) -> Tensor[B,H,W,2]
upsample_residual_flow(flow_lr, target_h, target_w, align_corners) -> Tensor[B,H,W,2]
fuse_grids(param_grid, residual_flow) -> Tensor[B,H,W,2]
warp_image(image, grid, mode, padding_mode, align_corners) -> Tensor[B,C,H,W]
jacobian_stats(grid) -> dict
```

## Geometry Contract Details
- Internal geometry tensor layout policy:
  - All geometry grid/displacement tensors inside Person 1 geometry functions use BHWC convention: `[B, H, W, 2]`.
  - Last dimension ordering is `(x, y)` / `(gx, gy)`.
- `params`:
  - Type: `torch.Tensor`.
  - Canonical shape: `[B, 8]`.
  - Canonical ordering (fixed): `[k1, k2, k3, p1, p2, dcx, dcy, s]`.
  - Names are fixed and position-dependent per the ordering above.
  - Optional `aspect` is out of Phase 0 contract and not part of the `[B, 8]` schema.
- `build_parametric_grid(...)`:
  - Output shape: `[B, H, W, 2]` where last dim is `(gx, gy)`.
  - Output is a backward sampling grid directly consumable by `torch.nn.functional.grid_sample`.
- `flow_lr`:
  - Canonical internal shape: `[B, H, W, 2]` (BHWC).
  - Canonical internal units: normalized grid delta in `grid_sample` coordinate space.
  - Must be upsampled to target resolution with the same global `align_corners=True`.
- `fuse_grids(param_grid, residual_flow)`:
  - Must perform `G_final = G_param + Delta_G_residual`.
  - Returns a single fused sampling grid for one-pass warping.
- `warp_image(image, grid, mode, padding_mode, align_corners)`:
  - `image` shape: `[B, C, H, W]`.
  - `grid` shape: `[B, H, W, 2]`.
  - `padding_mode` default: `"border"`.
- `jacobian_stats(grid)` must return a dictionary containing at least:
  - `negative_det_pct`
  - `det_min`
  - `det_p01`
  - `det_mean`
  - Optional: high-gradient area fraction metric.

## Inference API (Person 1)
Inference consumes model outputs from Person 2 and guarantees full-resolution, safety-gated, single-pass warping.

### Model Output Integration Contract
Model forward output dictionary keys and schema:
- `params` (required):
  - Type: `torch.Tensor`
  - Shape: `[B, 8]`
  - Ordering: `[k1, k2, k3, p1, p2, dcx, dcy, s]`
- `residual_flow` (optional, required for hybrid mode):
  - Accepted input layouts from model: BCHW (`[B, 2, Hr, Wr]`) or BHWC (`[B, Hr, Wr, 2]`)
  - Units at model output: pixel displacement (not normalized)
  - Reference resolution for pixel units: the residual tensor's own spatial resolution `(Hr, Wr)`
  - Axis order in 2-vector: `(dx, dy)` in image x/y directions

### Predictor Contract (`predictor.py`)
`predict(...)` behavior contract:
1. Load model/checkpoint in eval mode.
2. Preprocess image to model input size.
3. Run model forward pass.
4. Build parametric grid.
5. If residual exists, upsample + fuse into one final grid.
6. Compute safety metrics.
7. Apply fallback hierarchy if unsafe.
8. Warp full-resolution image exactly once.
9. Return corrected image plus metadata.

Required metadata keys:
- `mode_used` (`"hybrid" | "param_only" | "param_only_conservative"`)
- `safety` (structured safety result)
- `jacobian` (jacobian stats dictionary)
- `warnings` (list of warning strings; may be empty)

### Safety Contract (`safety.py`)
`evaluate_safety(...)` must return:
```python
{
  "safe": bool,
  "reasons": [str],
  "metrics": {...}
}
```
Minimum metrics tracked:
- out-of-bounds sample ratio
- invalid/black border ratio
- Jacobian foldover metrics
- residual magnitude summary

### Fallback Contract (`fallback.py`)
Required fallback order:
1. Hybrid (`params + residual`)
2. Param-only
3. Conservative param-only (stronger clamp and/or tighter zoom)

If all fail, emit best-effort output and set a hard warning flag in metadata.

### Writer Contract (`writer.py`)
- Save deterministic JPEG outputs for submission.
- Validate write success and output dimensions.
- Preserve filename mapping required by QA/packaging.
- Contract constants for deterministic JPEG writing:
  - `JPEG_QUALITY = 95`
  - `JPEG_SUBSAMPLING = "4:4:4"` (no chroma subsampling)
  - `JPEG_COLORSPACE = "RGB"` (encode from RGB, no implicit BGR path)
  - `JPEG_OPTIMIZE = False`
  - `JPEG_PROGRESSIVE = False`

## Residual Adapter Rules (Mandatory)
- Adapter accepts model `residual_flow` in BCHW or BHWC only.
- Adapter converts any accepted layout to canonical BHWC `[B, Hr, Wr, 2]`.
- Adapter converts units from pixel displacement to normalized grid delta before fusion.
- Pixel-to-normalized conversion must use `align_corners=True` formulas:
  - `dx_norm = 2 * dx_px / (Wr - 1)` when `Wr > 1`, else `0`
  - `dy_norm = 2 * dy_px / (Hr - 1)` when `Hr > 1`, else `0`
- After conversion, normalized residual may be upsampled to target `[H, W]` with bilinear interpolation and `align_corners=True`.
- Fusion is performed only in canonical BHWC normalized space.

## Versioning / Breaking Changes
Any change to the following is a breaking contract and requires explicit coordination and sign-off across Person 1, Person 2, and Person 3:
- Geometry API function signatures
- Global `align_corners` policy
- Tensor layout conventions (BCHW/BHWC) for geometry and model outputs
