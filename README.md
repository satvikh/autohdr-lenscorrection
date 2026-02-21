# AutoHDR Lens Correction

Geometry-first automatic lens distortion correction pipeline for hackathon-style leaderboard optimization.

## Objective
Build a robust system that corrects lens distortion by predicting a warp, not by generating new pixels.

From the planning docs, the end-to-end strategy is:
1. Predict global Brown-Conrady-style lens parameters.
2. Optionally predict a bounded low-resolution residual flow for local cleanup.
3. Fuse into one backward sampling grid.
4. Warp the original image once at full resolution.
5. Apply safety checks and deterministic fallbacks before submission packaging.

The score emphasis is geometry-heavy:
- Edge similarity: 40%
- Line straightness: 22%
- Gradient orientation similarity: 18%
- SSIM: 15%
- Pixel accuracy: 5%

## Team Concurrency Plan (Final)
- Person 1 (Satvik): Geometry + Inference + Safety
- Person 2 (Kush): Model + Losses + Training
- Person 3 (Suhaas): Data + Proxy Scorer + QA + Submission Tooling

This split is enforced through directory ownership, strict interface contracts, and scheduled integration checkpoints.

## Current Repository Status
Implemented now:
- Geometry coordinate conversions and identity grid generation.
- Parametric warp grid builder with bounded parameters.
- One-pass warp operator wrapper around `grid_sample`.
- Minimal param-only inference predictor.
- Deterministic JPEG writer.
- Unit and integration tests for geometry and baseline inference.

Planned but not yet implemented in code:
- Residual flow fusion and Jacobian utilities.
- Full safety/fallback modules for hybrid inference.
- Model/loss/training stack.
- Data pipeline, proxy metrics, QA tooling, submission packager.
- Ablation runner and experiment logging framework.

## Quick Start
### 1. Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision pillow pytest pyyaml scikit-image tqdm
```

### 2. Run tests
```bash
pytest -q
```

### 3. Run baseline inference script
```bash
python scripts/infer_test.py <input_dir> <output_dir>
```

## Repository Structure
```text
autohdr-lenscorrection/
  src/
    geometry/
    inference/
    models/        # planned
    losses/        # planned
    train/         # planned
    data/          # planned
    metrics/       # planned
    qa/            # planned
  scripts/
  tests/
  docs/
    README.md
    DOC_INDEX.md
    contracts.md
    owners.md
    OWNERSHIP_AND_CONCURRENCY.md
    INTEGRATION_CHECKPOINTS.md
    CODING_AGENT_PLAYBOOK.md
    satvik(person1)/
    kush(person2)/
    suhaas(person3)/
  docs_suhaas/  (legacy migration pointer only)
```

## Core Conventions (Non-Negotiable)
- Backward warp only.
- Geometry grid format: `BHWC` with last dim `(x, y)`.
- `align_corners=True` globally for geometry and resizing paths.
- Single fused warp pass in inference output path.
- Default padding mode: `border`.

## Collaboration Rules
- One owner per subsystem.
- Small PRs only.
- No direct commits to `main`.
- Shared interface changes must be documented in `docs/contracts.md`.
- Coding agents must be scoped to owned directories only.

## Documentation Map
- `AGENTS.md`: repo-wide contributor and coding-agent rules.
- `docs/DOC_INDEX.md`: central index for all project docs.
- `docs/contracts.md`: frozen interfaces across P1/P2/P3.
- `docs/owners.md`: explicit file and directory ownership.
- `docs/OWNERSHIP_AND_CONCURRENCY.md`: git/branch/merge and conflict prevention rules.
- `docs/INTEGRATION_CHECKPOINTS.md`: phase checkpoints and verification gates.
- `docs/CODING_AGENT_PLAYBOOK.md`: safe delegation templates for each person.
- `docs/satvik(person1)/`: person-1 execution docs.
- `docs/kush(person2)/`: person-2 execution docs.
- `docs/suhaas(person3)/`: person-3 execution docs.

## Immediate Next Priorities
1. Implement residual fusion and Jacobian modules (P1).
2. Stand up model/loss/training skeleton (P2).
3. Build dataset audit, splits, and proxy baseline (P3).
4. Reach Phase 1 integration checkpoint with contract-stable outputs.

## Notes
- `requirements.txt` is currently minimal and will need expansion as modules land.
- Always treat `docs/contracts.md` and `docs/owners.md` as source-of-truth for integration.