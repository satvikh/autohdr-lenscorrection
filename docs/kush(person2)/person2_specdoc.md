# Person 2 Spec Doc: Model + Losses + Training Lead

## Purpose
Define Person 2 execution scope, interfaces, and acceptance criteria for the training system.

## Primary Objective
Train a stable model that predicts geometry-correct corrections and improves proxy/external score.

## Out of Scope
- Geometry internals and inference safety/fallback implementation (Person 1).
- Dataset audit, proxy metric ownership, QA/submission tooling (Person 3).

## Required Contracts
Person 2 must comply with `docs/contracts.md` for:
- model output dict format
- params ordering and bounds semantics
- residual flow units/layout

## Model Requirements
### Output keys
- required: `params`
- optional: `residual_flow`, `pred_image`, `param_grid`, `final_grid`

### Output constraints
- `params`: shape `[B,8]`, ordered `[k1,k2,k3,p1,p2,dcx,dcy,s]`
- `residual_flow`: BCHW or BHWC pixel displacement
- residual head starts near zero to preserve early stability

## Loss Stack by Stage
### Stage 1 (parametric-only)
- pixel loss
- SSIM loss
- edge/gradient structure losses

### Stage 2 (hybrid)
- stage-1 losses plus:
- residual smoothness (TV)
- residual magnitude penalty
- Jacobian/foldover penalty

### Stage 3 (fine-tuning)
- increase geometry-focused terms
- preserve safety regularizers
- track proxy components, not only aggregate

## Training System Responsibilities
- robust dataloader/model/optimizer wiring
- AMP and gradient clipping support
- checkpointing and resume behavior
- validation pass with proxy hooks
- run metadata logging

## Acceptance Criteria
1. Stage 1 train/val executes end-to-end.
2. Stage 2 hybrid training executes without instability.
3. Loss values remain finite and trend sensibly.
4. Best checkpoints are selected by proxy score and logged.

## Risks and Mitigations
Risk: interface drift with P1/P3.
- Mitigation: strict contract adherence + checkpoint freeze.

Risk: residual flow dominates and causes artifacts.
- Mitigation: zero-init, magnitude bounds, regularization.

Risk: training instability.
- Mitigation: conservative LR, gradient clipping, anomaly logging.

## Deliverables
- model modules under `src/models/*`
- losses under `src/losses/*`
- training modules under `src/train/*`
- stage scripts and associated tests