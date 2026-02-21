# Project Charter

## Problem
Lens distortion bends geometry in images. The target task is to output corrected images that match hidden ground truth under a geometry-heavy metric.

## Why This Project Is Geometry-First
Success is dominated by alignment quality (edges, lines, gradient directions), not generative realism.

## Final Team Operating Model
- Person 1: Geometry + Inference + Safety
- Person 2: Model + Losses + Training
- Person 3: Data + Proxy Scorer + QA + Submission Tooling

The project is designed for concurrent execution through strict module ownership and interface contracts.

## Scope
In scope:
- Parametric lens warp prediction.
- Optional bounded residual flow refinement.
- Full-resolution one-pass warping.
- Safety checks, fallback logic, and submission QA.
- Multi-stage training and ablation-driven iteration.

Out of scope:
- Generative image synthesis approaches.
- Multi-pass warping in final inference output path.
- Geometric augmentations that break paired mapping.

## Success Criteria
Technical success:
- Contract-consistent dataset/model/geometry/proxy interfaces.
- Stable stage-1 baseline.
- Hybrid model outperforms baseline on proxy score without safety regressions.
- Full test-set inference and packaging pipeline reliability.

Competition success:
- No catastrophic hard-fail outputs in submission.
- Reproducible submission lineage (checkpoint + config + manifest + commit).
- Controlled improvement loop from ablations and proxy calibration.

## Constraints
- Preserve image geometry and output dimensions.
- Favor robust systems over risky late-stage complexity.
- Keep contracts stable across contributors and coding agents.
- Enforce single coordinate convention globally.