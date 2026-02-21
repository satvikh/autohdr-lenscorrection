# Person 3 Execution Checklist

Use this checklist before merging a Person 3 PR.

## Scope and Ownership
- [ ] Changes limited to Person 3 owned files.
- [ ] Shared file edits were coordinated.

## Contract Compliance
- [ ] Dataset sample keys match contract.
- [ ] Proxy scorer return schema unchanged or contract updated.
- [ ] QA output/manifests remain compatible.

## Validation
- [ ] Proxy metric tests pass.
- [ ] QA checks pass on sample inputs.
- [ ] Packaging script tested on representative output set.

## Quality Gates
- [ ] GT-vs-GT proxy behavior is near-perfect.
- [ ] Distorted-vs-GT proxy is appropriately worse.
- [ ] Hard-fail checks are triggered on synthetic bad cases.

## Handoff
- [ ] PR includes report snippets and expected artifact outputs.