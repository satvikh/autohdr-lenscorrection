# AutoHDR Lens Correction

Geometry-first automatic lens correction pipeline focused on leaderboard-safe outputs, fast iteration, and reproducible submissions.

## TL;DR
AutoHDR treats lens correction as a geometry problem first, not an image-generation problem. The system predicts a safe global correction, adds bounded local refinement only when useful, and always renders final outputs with a single full-resolution backward warp plus safety/fallback checks.

## The Problem We Are Solving
Lens distortion bends straight structures, shifts local geometry, and can create border artifacts. In this challenge setting, the main objective is geometric correctness: straighten what should be straight, preserve structure, and avoid catastrophic failures.

This means the project has to do more than produce visually pleasant outputs. It must consistently produce geometrically valid corrections across diverse scenes, while staying stable enough to trust in a submission pipeline.

## Why This Challenge Is Hard
A few factors make this non-trivial:

- Distortion contains both global behavior (whole-frame lens effects) and local behavior (scene-dependent corrections).
- Over-correcting can create foldovers, invalid sampling regions, and severe border damage.
- Under-correcting leaves measurable geometric error on the table.
- Methods that look good visually can still score poorly if geometry is wrong.
- Competition workflows demand repeatability: deterministic outputs, stable metrics, and clear experiment lineage.

So the challenge is not just to correct the image. It is to correct the geometry safely, measurably, and reproducibly.

## Our Three-Step Solution
### Step 1: Learn a strong global correction baseline
Start with a conservative global lens model that handles the dominant distortion reliably. This creates a stable foundation and prevents early overfitting to noisy local behavior.

### Step 2: Add bounded local refinement
Once the global model is stable, add a low-resolution residual correction branch for local improvements. Keep it constrained and regularized so local refinements help without destabilizing global geometry.

### Step 3: Metric-focused hardening and safety gating
Fine-tune for metric-relevant geometry quality, then enforce runtime safety checks and fallback routing so risky predictions degrade gracefully instead of failing catastrophically.

## Why This Strategy Works Best
This approach matches the structure of the real problem:

- It separates must-be-correct global geometry from nice-to-improve local refinement.
- It keeps the final rendering path simple and robust.
- It provides safety controls when predictions are uncertain.
- It supports fast iteration loops with measurable progress at each stage.
- It produces submission artifacts with traceable lineage and QA guarantees.

In short: strong baseline first, controlled complexity second, and safety/reproducibility throughout.

## Technical Architecture

### End-to-End Inference Path
1. Load a distorted RGB input.
2. Run the model to predict global parameters and optional residual flow.
3. Build a full-resolution parametric backward grid.
4. Convert residual flow into normalized grid-space deltas.
5. Upsample residual deltas to full resolution.
6. Fuse parametric and residual components into one final grid.
7. Evaluate safety metrics.
8. Apply fallback policy if unsafe (`hybrid -> param_only -> param_only_conservative`).
9. Run one full-resolution `grid_sample` pass.
10. Save deterministic JPEG + run metadata.

### Training System Shape
- Stage 1 (`stage1_param_only`): global parametric baseline.
- Stage 2 (`stage2_hybrid`): enable residual branch with flow and Jacobian regularization.
- Stage 3 (`stage3_finetune`): metric-focused refinement under safety-aware constraints.

Each stage is driven by three configs:
- model config (`configs/model/*.yaml`)
- loss config (`configs/loss/*.yaml`)
- train/engine config (`configs/train/*.yaml`)

### Runtime Safety and Fallback
Safety checks include:
- out-of-bounds sampling ratio
- invalid border ratio
- Jacobian foldover indicators
- residual magnitude diagnostics

If a candidate warp fails safety:
1. fall back from hybrid to param-only
2. if needed, clamp to conservative param-only
3. if still unsafe, emit best-effort output with warning metadata

## Core Contracts and Conventions
Authoritative source: `docs/contracts.md`.

Global invariants:
- Backward warp only.
- Geometry tensors in geometry modules use `BHWC`.
- Coordinate order is `(x, y)`.
- `align_corners=True` everywhere in geometry/interpolation.
- Final output generation is one-pass fused warp (no sequential production warps).

Dataset sample contract:
```python
{
  "input_image": Tensor[C,H,W],
  "target_image": Tensor[C,H,W],
  "image_id": str,
  "orig_size": tuple[int, int],
  "metadata": dict  # optional
}
```

Model output contract:
- required: `params` with canonical ordering `[k1,k2,k3,p1,p2,dcx,dcy,s]`
- optional: `residual_flow`, `pred_image`, `param_grid`, `final_grid`

Inference metadata contract (required keys):
- `mode_used`
- `safety`
- `jacobian`
- `warnings`

Proxy scorer contract:
- entrypoint: `compute_proxy_score(pred, gt, config) -> dict`
- includes total score, sub-scores, and hard-fail flags

## Repository Layout

### Core Code
- `src/geometry/*`: coordinates, parametric warp, residual fusion, warp ops, Jacobian stats
- `src/inference/*`: predictor, safety checks, fallback policy, writer
- `src/models/*`: backbone + parametric/residual heads + hybrid model
- `src/losses/*`: composite and component losses
- `src/train/*`: training engine, configs, optimizer/scheduler, checkpointing
- `src/data/*`: datasets, transforms, real-data loaders
- `src/metrics/*`: proxy metric components and aggregation
- `src/qa/*`: filename and image-integrity checks

### Scripts
- `scripts/infer_test.py`
- `scripts/train_stage1.py`
- `scripts/train_stage2.py`
- `scripts/train_stage3.py`
- `scripts/audit_dataset.py`
- `scripts/make_splits.py`
- `scripts/validate_proxy.py`
- `scripts/build_submission_zip.py`

### Tests
Run full suite: `pytest -q`

## Setup

### 1) Create environment
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

### 3) Verify installation
```bash
python -V
pytest -q
```

## Data Preparation

### Audit paired dataset
```bash
python scripts/audit_dataset.py \
  --original_dir <path/to/original_images> \
  --generated_dir <path/to/corrected_images> \
  --out_csv data/metadata/dataset_audit.csv
```

### Build stratified train/val splits
```bash
python scripts/make_splits.py \
  --audit_csv data/metadata/dataset_audit.csv \
  --out_dir data/splits/real_pairs \
  --val_frac 0.2 \
  --seed 123
```

## Training

### Stage commands
```bash
python scripts/train_stage1.py
python scripts/train_stage2.py
python scripts/train_stage3.py
```

### Common training flags
Each stage script supports:
- `--model-config`
- `--loss-config`
- `--train-config`
- `--run-name`
- `--use-synthetic`
- `--loader-module`
- `--loader-fn`
- `--resume-from`
- `--init-from`
- `--validate-only`
- `--warp-backend` (`person1` or `mock`)

Get full help:
```bash
python scripts/train_stage1.py --help
```

### Defaults and checkpoint flow
- Stage 1 defaults to `configs/*/stage1_param_only.yaml`
- Stage 2 defaults to `configs/*/stage2_hybrid.yaml` and initializes from `outputs/runs/stage1_param_only/best.pt`
- Stage 3 defaults to `configs/*/stage3_finetune.yaml` and initializes from `outputs/runs/stage2_hybrid/best.pt`

Trainer outputs under `outputs/runs/<run_name>/`:
- `best.pt`
- `last.pt`
- `resolved_config.json`

### Real-data loader environment variables
The default real loader entrypoint is `src.data.real_loader.build_train_val_loaders`.

Key environment variables:
- `AUTOHDR_DATA_ROOT` (dataset root)
- `AUTOHDR_SPLIT_DIR` (train/val split output or lookup directory)
- `AUTOHDR_TRAIN_METADATA` (optional metadata file)
- `AUTOHDR_VAL_FRAC`
- `AUTOHDR_SPLIT_SEED`
- `AUTOHDR_MAX_PAIRS`
- `AUTOHDR_BATCH_SIZE`
- `AUTOHDR_NUM_WORKERS`
- `AUTOHDR_RESIZE_HW` (format: `H,W`, empty disables resize)
- `AUTOHDR_TRAIN_HFLIP`
- `AUTOHDR_PIN_MEMORY`
- `AUTOHDR_REBUILD_SPLITS`

Example:
```powershell
$env:AUTOHDR_DATA_ROOT = "automatic-lens-correction/lens-correction-train-cleaned"
$env:AUTOHDR_SPLIT_DIR = "data/splits/real_pairs"
$env:AUTOHDR_BATCH_SIZE = "4"
$env:AUTOHDR_NUM_WORKERS = "0"
python scripts/train_stage1.py
```

## Inference

### Quick baseline (neutral stub)
```bash
python scripts/infer_test.py <input_dir> <output_dir>
```

### Checkpoint-backed inference
```bash
python scripts/infer_test.py <input_dir> <output_dir> \
  --checkpoint-id stage3_candidate \
  --checkpoint-path outputs/runs/stage3_finetune/best.pt \
  --model-config configs/model/resnet34_baseline.yaml \
  --device cuda \
  --config-path configs/train/stage3_finetune.yaml
```

Output:
- one JPEG per input image
- `run_metadata.json` with mode counts, fallback stats, and lineage metadata

## Proxy Validation and QA

### Validate proxy metrics on a split
```bash
python scripts/validate_proxy.py \
  --pred_dir <pred_dir> \
  --split_csv <split_csv> \
  --gt_root <optional_gt_root> \
  --config <optional_yaml_or_json> \
  --out_dir reports/validate_proxy
```

Artifacts:
- `per_image_scores.csv`
- `summary.json`

Exit behavior:
- returns `0` when fail rate is within allowed threshold
- returns `2` when fail rate exceeds allowed threshold

## Submission Packaging

### Build submission zip with QA gates
```bash
python scripts/build_submission_zip.py \
  --pred_dir <pred_dir> \
  --split_csv <split_csv> \
  --out_zip outputs/submissions/submission_v01.zip \
  --strict
```

Alternative required-id source:
```bash
python scripts/build_submission_zip.py \
  --pred_dir <pred_dir> \
  --ids_file <ids.txt|ids.csv|ids.json> \
  --out_zip outputs/submissions/submission_v01.zip
```

Additional options:
- `--no-strict` to allow zip creation even when QA fails
- `--force_zip` to force zip creation in strict mode failures
- `--config` to supply submission/image rules

Artifacts:
- submission zip (`--out_zip`)
- sibling QA report: `submission_qa.json`

## Tests and Validation

Run all tests:
```bash
pytest -q
```

Useful targeted runs:
```bash
pytest -q tests/test_contracts_integration.py
pytest -q tests/test_inference_pipeline.py tests/test_safety_fallback.py
pytest -q tests/test_proxy_metrics.py tests/test_qa_checks.py
pytest -q tests/test_stage_scripts_integration.py
```

## Reproducibility Checklist
For every experiment or submission, record:
- run name and stage
- commit hash
- model/loss/train config paths
- config overrides
- checkpoint path (`best.pt` used for candidate)
- proxy sub-scores and total
- safety/fallback statistics
- QA report status
- output artifact paths

Recommended practice:
- keep `main` stable
- integrate on `dev`
- freeze interfaces at checkpoints
- never silently change contracts

## Ownership and Collaboration Model
- Person 1: geometry + inference + safety
- Person 2: model + losses + training
- Person 3: data + proxy + QA + submission tooling

Authoritative ownership docs:
- `docs/owners.md`
- `docs/OWNERSHIP_AND_CONCURRENCY.md`

For shared high-conflict files (`docs/contracts.md`, `docs/owners.md`, `configs/*`, `requirements.txt`, `README.md`), coordinate changes explicitly.

## Definition of Done
- P1: geometry tests pass, full-res inference path active, safety/fallback active
- P2: stage 1/2/3 training stable, best checkpoints tracked
- P3: dataset audit/splits stable, proxy reliable, QA/submission tooling stable
- Team: reproducible, submission-ready pipeline with documented lineage

## Documentation Map
Read in this order:
1. `README.md`
2. `docs/DOC_INDEX.md`
3. `docs/contracts.md`
4. `docs/owners.md`
5. `docs/OWNERSHIP_AND_CONCURRENCY.md`
6. `docs/INTEGRATION_CHECKPOINTS.md`

Then follow role-specific docs under:
- `docs/satvik(person1)/`
- `docs/kush(person2)/`
- `docs/suhaas(person3)/`

## Notes
Cloud handoff configs/logs are available under:
- `autohdr_handoff/handoff/cloud_h100_20260224_075222/configs`
- `autohdr_handoff/handoff/cloud_h100_20260224_075222/logs`
