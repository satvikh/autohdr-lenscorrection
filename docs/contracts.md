# Contracts (Global, Cross-Role)

This document freezes interfaces between Person 1, Person 2, and Person 3 systems.

## Scope
This contract governs:
- dataset output format (Person 3 owned)
- model forward output format (Person 2 owned)
- geometry API and inference behavior (Person 1 owned)
- proxy scorer API (Person 3 owned)
- shared config key policy (shared, owner assigned)

## Global Conventions
- Warp direction: backward warp only.
- Grid coordinate convention: PyTorch `grid_sample` normalized coordinates in `[-1, 1]`.
- Global geometry `align_corners`: always `True`.
- Final output path: one-pass fused warp only.
- Default warp padding mode: `border`.

## Contract A: Dataset Sample Format (Owner: Person 3)
Each dataset sample must provide:
```python
{
  "input_image": Tensor[C,H,W],
  "target_image": Tensor[C,H,W],
  "image_id": str,
  "orig_size": tuple[int, int],
  "metadata": dict  # optional
}
```

Notes:
- `input_image` is distorted source.
- `target_image` is corrected ground truth.
- `orig_size` is source resolution before resize.
- Additional metadata keys are allowed but must not break required keys.

## Contract B: Model Output Format (Owner: Person 2)
Model forward output must be a dict with fixed keys:
- `params` (required): `Tensor[B,8]` ordered `[k1,k2,k3,p1,p2,dcx,dcy,s]`
- `residual_flow` (optional): `Tensor[B,2,Hr,Wr]` or `Tensor[B,Hr,Wr,2]`
- `pred_image` (optional): `Tensor[B,C,H,W]` when model applies geometry internally
- `param_grid` (optional debug): `Tensor[B,H,W,2]`
- `final_grid` (optional debug): `Tensor[B,H,W,2]`

Residual units:
- `residual_flow` is in pixel displacement units `(dx, dy)` relative to its own spatial resolution.

## Contract C: Geometry API (Owner: Person 1)
Mandatory signatures:
```python
build_parametric_grid(params, height, width, align_corners, device, dtype) -> Tensor[B,H,W,2]
upsample_residual_flow(flow_lr, target_h, target_w, align_corners) -> Tensor[B,H,W,2]
fuse_grids(param_grid, residual_flow) -> Tensor[B,H,W,2]
warp_image(image, grid, mode, padding_mode, align_corners) -> Tensor[B,C,H,W]
jacobian_stats(grid) -> dict
```

Geometry layout policy:
- Geometry grid and displacement tensors use `BHWC` with `(x, y)` ordering.

Parameter policy:
- Canonical `params` ordering is fixed and position-dependent.
- Optional `aspect` is not part of base `[B,8]` contract unless explicitly enabled by contract revision.

Residual adapter rules:
- Accept model `residual_flow` in BCHW or BHWC.
- Convert to canonical BHWC normalized grid delta before fusion.
- Conversion with `align_corners=True`:
  - `dx_norm = 2 * dx_px / (Wr - 1)` when `Wr > 1`, else `0`
  - `dy_norm = 2 * dy_px / (Hr - 1)` when `Hr > 1`, else `0`

## Contract D: Inference Behavior (Owner: Person 1)
`predict(...)` must perform:
1. load model/checkpoint in eval mode
2. preprocess to model size
3. forward pass
4. parametric grid build
5. residual upsample/fusion when available
6. safety evaluation
7. fallback routing when unsafe
8. full-resolution one-pass warp
9. metadata return

Required metadata keys:
- `mode_used`: `"hybrid" | "param_only" | "param_only_conservative"`
- `safety`: structured safety result
- `jacobian`: Jacobian stats dict
- `warnings`: list[str]

## Contract E: Proxy Scorer API (Owner: Person 3)
Required entry point:
```python
compute_proxy_score(pred, gt, config) -> dict
```

Minimum return keys:
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

## Contract F: Safety API (Owner: Person 1, consumed by Person 3)
`evaluate_safety(...)` returns:
```python
{
  "safe": bool,
  "reasons": [str],
  "metrics": {
    "oob_ratio": float,
    "border_invalid_ratio": float,
    "jacobian_negative_det_pct": float,
    "residual_magnitude": float
  }
}
```

## Contract G: Fallback Policy (Owner: Person 1)
Required order:
1. hybrid
2. param-only
3. conservative param-only

If all fail, emit best-effort output with hard warning metadata.

## Contract H: QA and Submission Interface (Owner: Person 3)
Submission tooling must guarantee:
- filename mapping correctness
- image decode integrity
- output dimension consistency
- deterministic zip packaging
- manifest containing checkpoint/config/date/commit metadata

## Contract I: Config Schema Ownership
Shared config keys include:
- image size
- interpolation mode
- `align_corners`
- parameter bounds
- loss weights
- batch size
- data paths

Ownership:
- primary owner: Person 3
- train-config review owner: Person 2
- geometry convention review: Person 1

## Breaking Change Policy
Any of the following requires coordinated sign-off from all three owners:
- model output dict key changes
- dataset sample key changes
- geometry API signature changes
- `align_corners` policy changes
- tensor layout changes
- proxy scorer return schema changes
- fallback mode semantics changes