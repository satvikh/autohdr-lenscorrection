# Person 1 Execution Checklist

Use this checklist before merging a Person 1 PR.

## Scope and Ownership
- [ ] Changes limited to Person 1 owned files.
- [ ] Shared file edits were pre-coordinated.

## Contract Compliance
- [ ] Geometry API signatures unchanged or contract updated.
- [ ] `align_corners=True` preserved in all paths.
- [ ] BHWC + `(x, y)` conventions preserved.

## Validation
- [ ] Geometry unit tests updated/passed.
- [ ] Inference integration tests updated/passed.
- [ ] Visual sanity checked on representative images.

## Safety/Fallback
- [ ] Safety metrics include required fields.
- [ ] Fallback order remains hybrid -> param-only -> conservative.
- [ ] Metadata contains mode/safety/jacobian/warnings keys.

## Handoff
- [ ] PR notes include files changed, tests, and interface impact.