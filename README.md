# AutoHDR Lens Correction

Geometry-first automatic lens correction system focused on leaderboard-safe outputs, reproducibility, and fast iteration.

This project corrects lens distortion by predicting:
1. A global parametric lens warp (Brown-Conrady style coefficients).
2. An optional low-resolution residual flow for local refinement.

Then it applies a single fused full-resolution backward warp and runs safety/fallback checks before writing deterministic outputs.

## Why This Exists
Lens distortion bends lines, edges, and local geometry. In this benchmark setting, success is mostly geometric alignment quality, not image synthesis quality.

This repo is designed for:
- Strong geometric correctness constraints.
- Safe inference behavior under bad predictions.
- Repeatable training and submission generation.

## What Makes It Non-Trivial
- Strict global geometry contracts:
  - Backward warp only.
  - `BHWC` geometry tensor layout.
  - Coordinate order `(x, y)`.
  - `align_corners=True` everywhere.
  - One-pass fused sampling in production inference.
- Stage-based training with different behavior (param-only then hybrid).
- Runtime safety checks + fallback hierarchy (`hybrid -> param_only -> param_only_conservative`).
- QA and deterministic submission packaging.

## High-Level Architecture
- `src/models/*`
  - Hybrid model with CNN backbone + parametric head + residual flow head.
- `src/geometry/*`
  - Coordinate transforms, parametric grid construction, residual fusion, warp ops, Jacobian stats.
- `src/inference/*`
  - Predictor, safety evaluation, fallback logic, output writing.
- `src/losses/*`, `src/train/*`
  - Composite stage-aware losses, train/eval step logic, engine, schedulers, checkpointing.
- `src/data/*`, `src/metrics/*`, `src/qa/*`
  - Data pipeline, proxy scoring, QA checks, packaging support.
- `scripts/*`
  - Training entrypoints, inference, dataset audit/splits, proxy validation, zip build.

## Typical End-to-End Flow
1. Load distorted image.
2. Predict global params (+ optional residual flow).
3. Build full-res parametric backward grid.
4. Convert + upsample residual to normalized grid deltas.
5. Fuse into one final grid.
6. Run one full-res `grid_sample`.
7. Evaluate safety metrics.
8. Apply fallback if needed.
9. Save corrected image + run metadata.

## Quick Setup
### 1) Create and activate environment
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If your environment is missing core packages used in tests/scripts, install:
```bash
pip install torch torchvision pillow pytest pyyaml scikit-image tqdm
```

## Quick Validation
Run the test suite:
```bash
pytest -q
```

## Quick Inference
Run directory inference using the default neutral stub model:
```bash
python scripts/infer_test.py <input_dir> <output_dir>
```

Example:
```bash
python scripts/infer_test.py outputs/integration/real_test_subset outputs/integration/real_test_pred
```

Useful optional args:
- `--checkpoint-id <name>`
- `--checkpoint-path <path/to/best.pt>`
- `--model-config <path/to/model.yaml>`
- `--device cpu|cuda`
- `--config-path <path/to/train-config-or-run-config>`

## Training Entrypoints
- Stage 1:
```bash
python scripts/train_stage1.py
```
- Stage 2:
```bash
python scripts/train_stage2.py
```
- Stage 3:
```bash
python scripts/train_stage3.py
```

Each stage script supports config overrides; inspect options with:
```bash
python scripts/train_stage1.py --help
```

## Proxy / QA / Submission Tooling
Validate proxy metrics for a prediction directory:
```bash
python scripts/validate_proxy.py --pred_dir <pred_dir> --split_csv <split_csv> --out_dir <out_dir>
```

Build submission zip with strict QA gating:
```bash
python scripts/build_submission_zip.py --pred_dir <pred_dir> --split_csv <split_csv> --out_zip <out_zip> --strict
```

## Key Contracts and Conventions
Authoritative contracts live in `docs/contracts.md`.

Most important invariants:
- Backward warp only.
- `align_corners=True` for geometry and interpolation.
- One-pass fused output warp.
- Stable model/dataset/proxy interfaces across contributors.

## Where to Read Next
1. `docs/DOC_INDEX.md`
2. `docs/contracts.md`
3. `docs/owners.md`
4. `docs/ARCHITECTURE.md`
5. `docs/OPERATIONS_RUNBOOK.md`

## Notes
- For exact cloud handoff runs/config snapshots, see:
  - `autohdr_handoff/handoff/cloud_h100_20260224_075222/logs`
  - `autohdr_handoff/handoff/cloud_h100_20260224_075222/configs`
