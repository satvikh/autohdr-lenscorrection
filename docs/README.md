# Docs Directory Guide

This folder is the source of truth for project planning, contracts, ownership, and execution.

## Folder Contract
- `docs/*.md`: global project documentation used by everyone.
- `docs/satvik(person1)/*.md`: person-1 scoped docs.
- `docs/kush(person2)/*.md`: person-2 scoped docs.
- `docs/suhaas(person3)/*.md`: person-3 scoped docs.

## Required Global Docs
- `docs/contracts.md`
- `docs/owners.md`
- `docs/OWNERSHIP_AND_CONCURRENCY.md`
- `docs/INTEGRATION_CHECKPOINTS.md`
- `docs/CODING_AGENT_PLAYBOOK.md`

## Editing Rules
- Keep cross-team decisions in global docs.
- Keep person folders focused on owned scope and work plans.
- For shared interfaces, update `docs/contracts.md` first.
- For ownership changes, update `docs/owners.md` and `docs/OWNERSHIP_AND_CONCURRENCY.md` in the same PR.

## New Contributor Onboarding
1. Read `README.md`.
2. Read `AGENTS.md`.
3. Read `docs/DOC_INDEX.md`.
4. Read global contracts/ownership docs.
5. Read your person folder README and spec/concurrency docs.
6. Work only in your owned paths unless coordinated.