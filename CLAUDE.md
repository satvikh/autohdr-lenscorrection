# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoHDR Lens Correction is a geometry-first automatic lens distortion correction pipeline. The system corrects lens distortion by predicting a warp transformation (not generating new pixels) using a Brown-Conrady parametric model with an optional low-resolution residual flow for local cleanup.

**Scoring weights:** Edge similarity (40%), line straightness (22%), gradient orientation (18%), SSIM (15%), pixel accuracy (5%).

## Setup

```bash
bash setup.sh   # Creates .venv and installs all dependencies
```

## Commands

```bash
# Run all tests
pytest -q

# Run a single test file
pytest tests/test_coords.py -v

# Run baseline inference
python scripts/infer_test.py <input_dir> <output_dir>

# Audit dataset for geometry features
python scripts/audit_dataset.py <source_dir> --output-csv splits.csv

# Generate train/val/test splits
python scripts/make_splits.py
```

`pytest.ini` sets `pythonpath = .` and discovers `test_*.py` files.

## Architecture

The system has three subsystems with explicit ownership:

- **Person 1 (Satvik):** `src/geometry/`, `src/inference/`, `scripts/infer_test.py`
- **Person 2 (Kush):** `src/models/`, `src/losses/`, `src/train/` (planned)
- **Person 3 (Suhaas):** `src/data/`, `src/metrics/`, `src/qa/`, `scripts/audit_dataset.py`, `scripts/make_splits.py`

### Inference Pipeline (10-step, `src/inference/predictor.py`)

1. Load distorted image
2. Optional resize to model input size
3. Forward pass → `params` (Tensor[B,8]) + optional `residual_flow`
4. Build parametric backward warp grid from Brown-Conrady params
5. Upsample/adapt residual flow to grid resolution (if available)
6. Fuse parametric grid + residual flow
7. Evaluate safety metrics (Jacobian, OOB, borders, residual magnitudes)
8. Apply fallback hierarchy if unsafe
9. Single-pass full-resolution `grid_sample` warp
10. Save deterministic JPEG + return metadata

Fallback hierarchy: **Hybrid** (param + residual) → **Param-only** → **Conservative param-only** (tighter clamps).

### Geometry Layer (`src/geometry/`)

- `coords.py`: Pixel ↔ normalized ↔ grid coordinate conversions (`align_corners=True`)
- `parametric_warp.py`: `build_parametric_grid()` — Brown-Conrady model with 8 params: `[k1, k2, k3, p1, p2, dcx, dcy, s]`
- `warp_ops.py`: Wraps `F.grid_sample` (BHWC grid, `align_corners=True`)
- `jacobian.py`: Finite-difference Jacobian determinant for safety checks
- `residual_fusion.py`: Upsample BCHW/BHWC residual flow → normalized grid delta → fuse

### Data Layer (`src/data/`)

`PairedLensDataset` reads a split CSV with columns `image_id`, `input_path`, `target_path`. Returns `{input_image: Tensor[C,H,W], target_image: Tensor[C,H,W], image_id: str, orig_size: tuple}` as float32 RGB in [0, 1].

## Key Conventions (Non-Negotiable)

- **Backward warp only** — sample source pixels from output coordinates
- **Geometry tensors:** BHWC layout with **(x, y)** coordinate order
- **`align_corners=True`** everywhere in PyTorch grid ops
- **Single fused warp pass** in production
- **Default padding:** `border`
- **JPEG output:** quality=95, subsampling=0 (4:4:4), non-progressive

## Parametric Bounds (Enforced via Clamping)

| Param | Range |
|-------|-------|
| k1 | [-0.6, 0.6] |
| k2 | [-0.3, 0.3] |
| k3 | [-0.15, 0.15] |
| p1, p2 | [-0.03, 0.03] |
| dcx, dcy | [-0.08, 0.08] |
| s | [0.90, 1.20] |

## Interface Contracts

Frozen contracts live in `docs/contracts.md`. Key model output format (Contract B):
```python
{
  "params": Tensor[B, 8],                         # required
  "residual_flow": Tensor[B,2,Hr,Wr] | Tensor[B,Hr,Wr,2],  # optional
  "pred_image": Tensor[B,C,H,W],                  # optional
}
```

**Breaking any contract requires sign-off from all 3 owners.**

## Ownership & Branching

- No direct commits to `main`; use PRs
- One owner per subsystem; cross-subsystem changes require reviewer approval
- See `docs/owners.md` for the explicit file ownership map
- See `docs/contracts.md` for all 9 frozen interface contracts (A–I)
