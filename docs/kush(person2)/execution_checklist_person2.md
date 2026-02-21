# Person 2 Execution Checklist

Use this checklist before merging a Person 2 PR.

## Scope and Ownership
- [ ] Changes limited to Person 2 owned files.
- [ ] Shared file edits were coordinated.

## Contract Compliance
- [ ] Output dict keys match `docs/contracts.md`.
- [ ] Params ordering remains fixed.
- [ ] Residual layout/units documented and unchanged.

## Validation
- [ ] Model output shape tests pass.
- [ ] Loss tests pass.
- [ ] Training smoke test completed.

## Stability
- [ ] No NaN/Inf losses.
- [ ] Gradient behavior monitored.
- [ ] Residual branch remains bounded.

## Handoff
- [ ] PR notes include checkpoint/config/test evidence and interface impact.