# Ownership and Concurrency Guide

This guide operationalizes the final 3-person concurrency plan.

## Core Principles
1. Split by systems, not by line count.
2. One owner per module.
3. Freeze interfaces early and at checkpoints.
4. Integrate frequently in small slices.
5. Keep coding agents scoped and deterministic.

## Final Role Split
- Person 1: Geometry + Inference + Safety
- Person 2: Model + Losses + Training
- Person 3: Data + Proxy Scorer + QA + Submission Tooling

## Branching Strategy
- `main`: stable only.
- `dev`: integration branch (recommended).
- feature branches:
  - `feat/geometry-*`
  - `feat/model-*`
  - `feat/data-proxy-qa-*`

Never work directly on `main`.

## Merge Strategy
- One integration captain merges checkpoint PRs.
- PRs must be small, scoped, and tested.
- Rebase or merge from `dev` at least every 1-2 hours in active sprint windows.

## One-Owner-One-File Rule
Even inside owned directories, avoid concurrent edits to the same file by multiple people.

## Shared File Coordination
High-conflict files:
- `docs/contracts.md`
- `docs/owners.md`
- `configs/*`
- `requirements.txt`

Policy:
1. Announce intent.
2. Assign one active editor.
3. Others use adapters/shims if blocked.
4. Merge quickly and notify team.

## Checkpoint-Based Interface Freeze
At each integration checkpoint, freeze:
- function signatures
- dict output keys
- config field names

Only break freeze for critical issues, and update docs immediately.

## Required Communication Cadence
- 15-minute sync every 3-4 hours.
- Integration checkpoint syncs on planned milestones.
- Shared status board fields:
  - active branch
  - current task
  - blockers
  - interface changes

## PR Checklist
- What changed.
- Files touched.
- How tested.
- Interface changes, if any.
- Visual evidence for geometry/inference changes where relevant.

## Handoff Template
- Branch:
- Commit(s):
- Files changed:
- Interface impact:
- Test evidence:
- Remaining risks/TODOs: