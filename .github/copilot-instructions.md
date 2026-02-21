# AI Coding Agent Instructions: autohdr-lenscorrection

## Project Overview
This is an **image lens distortion correction system** that predicts and applies optical aberration corrections to images using parametric (Brown-Conrady) lens models and optional hybrid ML-based predictions.

### Core Mission
1. **Geometry Module** (`src/geometry/`): Parametric lens distortion modeling + grid-based image warping
2. **Inference Pipeline** (`src/inference/`): Model integration + end-to-end prediction workflow
3. **Validation**: Comprehensive coordinate transform testing and distortion parameter handling

---

## Critical Architecture Patterns

### 1. Geometry Contract (Non-Negotiable)
**All geometry operations enforce `align_corners=True`** to maintain consistent coordinate mappings across transforms:

- **Coordinate Spaces**: `pixel ↔ normalized_image ↔ grid`
  - Pixel: `[0, width-1]` × `[0, height-1]` (image coords)
  - Normalized: `[-1, 1]` × `[-1, 1]` (PyTorch grid_sample standard)
  - Grid: `[B, H, W, 2]` BHWC format with (x, y) order
  
- **Transform Chain** (see `src/geometry/coords.py`):
  ```
  pixel_to_normalized_image() → normalized_image_to_grid() → grid_sample()
  ```
  Any new coordinate transform **must** roundtrip through tests with `torch.allclose(atol=1e-10)`

- **Edge Case**: Handle `height=1` or `width=1` gracefully (degenerate grids)

### 2. Parametric Distortion Model
**8 Brown-Conrady parameters** + optional aspect ratio (see `src/geometry/parametric_warp.py`):

```python
k1, k2, k3      # Radial distortion coefficients (3 terms)
p1, p2          # Tangential distortion (prism effect)
dcx, dcy        # Shift in principal point offset
s               # Scale correction
aspect (opt)    # Aspect ratio (ENABLE_ASPECT=False by default)
```

**Safety Ranges** (hard-clamped, non-negotiable for production):
- `k1: [-0.6, 0.6]`, `k2: [-0.3, 0.3]`, `k3: [-0.15, 0.15]`
- `p1, p2: [-0.03, 0.03]`, `dcx, dcy: [-0.08, 0.08]`
- `s: [0.90, 1.20]`, `aspect: [0.97, 1.03]`

**Feature Flag**: `ENABLE_ASPECT` gates the optional 9th parameter; default is False.

### 3. Inference Pipeline Flow
**`src/inference/predictor.py`** orchestrates the full pipeline:

1. **Load** → `_load_image_as_tensor()` → normalize to `[0,1]` float, shape `[B, C, H, W]`
2. **Resize** (optional) → `_maybe_resize()` using `align_corners=True`
3. **Model Call** → expects `dict[str, Tensor]` with required key `"params"` of shape `[B, 8+]`
4. **Grid Build** → `build_parametric_grid()` converts params to backward sampling grid
5. **Warp** → `warp_image()` applies `F.grid_sample()` with `align_corners=True`
6. **Return** → warped tensor + metadata dict

**Error Contract**: Model must return `{"params": Tensor[B, 8+]}` or prediction fails.

### 4. Output Serialization
**`src/inference/writer.py`** enforces deterministic JPEG export:

- **Format Constraints**: JPEG quality 95, 4:4:4 subsampling, no progressive encoding
- **Validation**: Save → Re-load → Verify dimensions match to catch codec issues
- **Channel Handling**: 1-channel → repeat to 3 RGB; 3-channel preserved; anything else fails

---

## Testing Philosophy

### Geometry Tests (`tests/test_coords.py`)
- **Roundtrip validation**: `pixel → normalized → pixel` must error < 1e-10
- **Batched & Degenerate**: Test multiple batch sizes, edge cases (1×5, 5×1 grids)
- **Grid packing**: Verify BHWC format and axis order correctness

### Pipeline Tests
- Validate parameter clamping respects safety ranges
- Check metadata propagation (input shape → output shape must match after warping)

### Key Principle
All tests use `torch.manual_seed()` for reproducibility and strict tolerance checks.

---

## Developer Workflows

### Setup
```bash
./setup.sh                    # Creates venv, installs requirements.txt
source venv/bin/activate      # Must activate before developing
```

### Running Tests
```bash
python -m pytest tests/                    # Run all tests
python -m pytest tests/test_coords.py -v  # Specific test file
```

### Running Inference
```bash
python scripts/infer_test.py <input_dir> <output_dir>
# Stub model emits neutral identity warp (all params = [0...0, 1.0])
# Output JPEG saved to output_dir with same stem names
```

---

## Project-Specific Conventions

### Naming & Paths
- Test prediction folders: `data/pred_test_param`, `data/pred_test_hybrid` 
- Output selection logic: Hybrid preferred if safe; fallback to parametric if hybrid fails
- File matching: Use stem-based matching for cross-folder validation (handle extension mismatches)

### Quality Constraints
- **No black borders or empty corners** in output (hard failure for leaderboard)
- **Edge similarity must not fall below minimum threshold** or score = 0.0
- **SSIM, gradient alignment, line straightness** all weighted in composite metric (see `docs_person_3/metric_spec.md`)

### Data Flow
- **Input**: Original RGB image + model-predicted distortion parameters
- **Process**: Build backward grid from params → grid_sample with border padding
- **Output**: Corrected RGB image at original resolution, valid decodable JPEG

---

## When Adding New Features

1. **Geometry changes**: Update `src/geometry/coords.py` + add roundtrip tests
2. **Parameter changes**: Update ranges in `parametric_warp.py` + clamp logic
3. **Model integration**: Ensure model returns `{"params": Tensor}` in `Predictor.__init__` validation
4. **Output changes**: Test JPEG serialization in `writer.py` with dimension validation

---

## Key Files by Function

| File | Purpose | Key Symbols |
|------|---------|-------------|
| `src/geometry/coords.py` | Coordinate transforms | `pixel_to_normalized_image()`, `normalized_image_to_grid()` |
| `src/geometry/parametric_warp.py` | Distortion model | `build_parametric_grid()`, param ranges |
| `src/geometry/warp_ops.py` | Grid sampling | `warp_image()` (wraps `F.grid_sample`) |
| `src/inference/predictor.py` | Pipeline orchestration | `Predictor.predict()`, model contract |
| `src/inference/writer.py` | Output serialization | `save_jpeg()` with validation |
| `scripts/infer_test.py` | CLI entry point | `NeutralModel`, batch processing loop |
