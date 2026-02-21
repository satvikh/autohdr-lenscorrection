# Integration Checkpoints

Checkpoint plan derived from the finalized concurrency strategy.

## Phase 0 Checkpoint (0-90 min)
Goal:
- Align on contracts, ownership, branch strategy, and repo scaffold.

Verify:
1. `docs/contracts.md` has dataset/model/geometry/proxy contracts.
2. `docs/owners.md` is complete.
3. Branches are created and ownership acknowledged.
4. `align_corners`, params ordering, and output dict keys are frozen.

## Phase 1 Checkpoint (~Hour 6)
Goal:
- Parallel foundations integrated.

Verify:
1. Dataset returns expected sample structure.
2. Model forward pass runs on real batch.
3. Geometry warps dummy params on real batch.
4. Shapes and conventions align with contracts.
5. No contract mismatches.

## Phase 2 Checkpoint (~Hour 12)
Goal:
- First end-to-end parametric-only train/val loop.

Verify:
1. Stage-1 train script runs on real data.
2. Proxy validation executes and logs sub-scores.
3. Visual outputs are generated and inspected.
4. No catastrophic borders/artifacts.

## Phase 3 Checkpoint (~Hour 20)
Goal:
- First external submission (safe baseline).

Verify:
1. Full-resolution inference runs on test set.
2. QA checks pass (filenames, integrity, dimensions).
3. Submission zip is accepted by external scorer.
4. Manifest records checkpoint/config/timestamp.

## Phase 4 Checkpoint (~Hour 30)
Goal:
- Hybrid model improves over baseline.

Verify:
1. Hybrid proxy score beats param-only baseline.
2. Safety/fallback rates are acceptable.
3. Best checkpoint is selected and logged.
4. Submission candidate set is review-approved.

## Phase 5 (Final Stretch)
Goal:
- Optional TTO and final submission hardening.

Verify:
1. TTO only retained when score improves.
2. Final QA pass clean.
3. Final zip + manifest + experiment log archived.