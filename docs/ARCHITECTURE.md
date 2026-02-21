# Architecture

## End-to-End Pipeline
1. Input distorted image.
2. Predict global parametric warp coefficients.
3. Optionally predict low-resolution residual flow.
4. Build full-resolution parametric backward grid.
5. Upsample/convert residual to normalized grid offsets.
6. Fuse to one final grid.
7. Run one `grid_sample` on original full-resolution image.
8. Evaluate safety metrics.
9. Apply fallback if unsafe.
10. Save deterministic JPEG and metadata.

## Role-Aligned System Boundaries
### Person 1 System
- Geometry core (`src/geometry/*`)
- Inference, safety, fallback (`src/inference/*`)

### Person 2 System
- Model outputs (`src/models/*`)
- Losses and training loops (`src/losses/*`, `src/train/*`)

### Person 3 System
- Data ingestion/splits (`src/data/*`)
- Proxy scoring (`src/metrics/*`)
- QA and submission tooling (`src/qa/*`, packaging scripts)

## Critical Interface Contracts
- Dataset sample contract (P3 -> P2)
- Model output dict contract (P2 -> P1/P3)
- Geometry API contract (P1 -> P2)
- Proxy scorer API contract (P3 -> P2)

Authoritative definitions live in `docs/contracts.md`.

## Geometry Conventions
- Backward sampling grid for PyTorch `grid_sample`.
- Normalized coordinates in `[-1, 1]`.
- Internal geometry tensors: `BHWC`.
- `align_corners=True` everywhere.
- Padding default: `border`.

## Safety and Failure Prevention
Runtime checks should cover:
- out-of-bounds sampling ratio
- border artifact ratio
- Jacobian foldover indicators
- residual magnitude limits

Fallback order:
1. hybrid
2. param-only
3. conservative param-only

## Design Principles
- Keep geometry deterministic and testable.
- Keep residual corrections bounded and smooth.
- Optimize for metric-aligned structure.
- Make every submission reproducible and auditable.