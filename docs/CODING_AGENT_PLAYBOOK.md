# Coding Agent Playbook

Use this playbook to delegate safely without causing merge conflicts.

## Universal Rules
1. Always provide explicit file targets.
2. Restrict to owned directories.
3. Require tests and usage notes in output.
4. Do not request repo-wide refactors.
5. Do not modify shared files unless explicitly tasked.

## Prompt Template
Use:
- Task:
- Allowed files:
- Forbidden files:
- Required tests:
- Contract constraints:
- Acceptance criteria:

## Person 1 Agent Delegation
Good targets:
- `src/geometry/coords.py`
- `src/geometry/parametric_warp.py`
- `src/geometry/jacobian.py`
- `src/inference/safety.py`
- geometry/inference tests

## Person 2 Agent Delegation
Good targets:
- `src/models/*`
- `src/losses/*`
- `src/train/*`
- `scripts/train_stage*.py`
- model/loss tests

## Person 3 Agent Delegation
Good targets:
- `src/data/*`
- `src/metrics/*`
- `src/qa/*`
- `scripts/audit_dataset.py`
- `scripts/validate_proxy.py`
- `scripts/build_submission_zip.py`

## Mandatory Human Review Checklist
- Tensor shapes and key ordering verified.
- Coordinate and `align_corners` conventions preserved.
- Dependencies are valid.
- No unrelated file edits.
- Tests pass locally or failures explained.