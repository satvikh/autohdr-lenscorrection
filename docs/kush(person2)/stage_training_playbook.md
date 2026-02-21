# Person 2 Stage Training Playbook

## Stage 1: Parametric Baseline
Goal:
- train global parametric correction safely and quickly.

Checklist:
- model outputs params only or ignores residual branch
- core image/structure losses enabled
- train/val loops stable
- first baseline checkpoints saved

## Stage 2: Hybrid Enablement
Goal:
- add bounded residual corrections without destabilizing geometry.

Checklist:
- residual head enabled and zero-initialized
- flow regularizers active
- Jacobian penalty active
- residual magnitude monitored

## Stage 3: Metric-Focused Fine-Tuning
Goal:
- improve edge/line/gradient proxy components while preserving safety.

Checklist:
- tuned loss weights documented
- checkpoint comparisons logged
- safety/failure rate reviewed with Person 1 and Person 3

## Required Logging for Every Run
- run id
- config path and overrides
- checkpoint parent
- train/val loss curves
- proxy sub-scores
- notes on artifacts/failures