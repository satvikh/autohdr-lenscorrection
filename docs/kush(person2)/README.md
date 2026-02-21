# Person 2 Docs (Kush)

Person 2 owns Model + Losses + Training.

## Ownership Scope
- Model architecture and output heads.
- Loss stack and regularizers.
- Stage-based training loops and optimization.
- Checkpoint tracking by proxy score.

## Owned Code Paths
- `src/models/*`
- `src/losses/*`
- `src/train/*`
- `scripts/train_stage1.py`
- `scripts/train_stage2.py`
- `scripts/train_stage3.py`
- `tests/test_model_outputs.py`
- `tests/test_losses.py`

## Core Documents
- `person2_specdoc.md`
- `concurrency_plan_person2.md`
- `stage_training_playbook.md`
- `execution_checklist_person2.md`

## Integration Dependencies
- Consumes dataset contract from Person 3.
- Consumes geometry API from Person 1.
- Must emit model output dict matching `docs/contracts.md`.