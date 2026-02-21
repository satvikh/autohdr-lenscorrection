# Person 2 Concurrency Plan (Model + Losses + Training)

## Mission
Build and train the hybrid model to predict stable, geometry-correct warps.

## Phase 0 (0-90 min)
- Lock model output dict contract in `docs/contracts.md`.
- Confirm parameter bounds and output keys.
- Align with Person 1 on early mock integration path.

## Phase 1 (Hours 1-6)
Build:
- `src/models/backbones.py`
- `src/models/coord_channels.py`
- `src/models/heads_parametric.py`
- `src/models/heads_residual.py`
- `src/models/hybrid_model.py` (skeleton)
- `src/train/engine.py` and `src/train/optim.py` (skeletons)
- `tests/test_model_outputs.py`

Done criteria:
- forward pass on dummy and real batches
- contract-aligned output keys and shapes
- residual head zero-init and bounded

## Phase 2 (Hours 6-12)
Build:
- `src/losses/pixel.py`
- `src/losses/ssim_loss.py`
- `src/losses/gradients.py`
- `src/losses/composite.py` (stage-1)
- `scripts/train_stage1.py`

Done criteria:
- stage-1 training runs end-to-end
- validation and logging execute
- no NaNs or exploding gradients

## Phase 3 (Hours 12-20)
Build:
- `src/losses/flow_regularizers.py`
- `src/losses/jacobian_loss.py`
- stage-2 composite updates
- `scripts/train_stage2.py`

Done criteria:
- hybrid training runs forward/backward
- residual penalties are finite and effective

## Phase 4 (Hours 20-30)
Build:
- stage-2/3 tuning and checkpoint ranking
- `scripts/train_stage3.py`

Done criteria:
- hybrid outperforms param-only on proxy
- best checkpoints tracked and tagged

## Phase 5 (Optional)
- integrate bounded TTO refinement path if stable and beneficial

## Integration Checklist at Each Checkpoint
1. Verify model outputs are contract-compliant.
2. Verify geometry integration works without patching P1 internals.
3. Verify proxy validation hooks consume outputs correctly.
4. Record checkpoint + config for submission traceability.