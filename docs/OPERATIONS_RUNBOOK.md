# Operations Runbook

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision pillow pytest pyyaml scikit-image tqdm
```

## Verify Setup
```bash
python -V
pytest -q
```

If test collection fails with `ModuleNotFoundError: torch`, install PyTorch in the active environment.

## Branch Setup
```bash
git checkout -b feat/<scope>
```

Recommended:
- P1: `feat/geometry-*`
- P2: `feat/model-*`
- P3: `feat/data-proxy-qa-*`

## Daily Workflow
1. Pull/rebase from `dev`.
2. Implement only inside owned directories.
3. Run targeted tests.
4. Open small PR with evidence.
5. Sync at checkpoint windows.

## Baseline Inference Command
```bash
python scripts/infer_test.py <input_dir> <output_dir>
```

Current script behavior:
- Uses neutral param-only stub model.
- Produces identity-style corrected outputs.
- Saves deterministic JPEGs.

## Stage Training Commands (Planned)
```bash
python scripts/train_stage1.py --config configs/stage1_parametric.yaml
python scripts/train_stage2.py --config configs/stage2_hybrid.yaml
python scripts/train_stage3.py --config configs/stage3_finetune.yaml
```

## Proxy and QA Commands (Planned)
```bash
python scripts/validate_proxy.py --pred <pred_dir> --gt <val_dir>
python scripts/build_submission_zip.py --pred <pred_dir> --out <zip_path>
```

## Integration Checkpoint Routine
At checkpoint time:
1. Pull latest `dev`.
2. Run interface smoke tests.
3. Verify contract keys and tensor shapes.
4. Resolve mismatches immediately.
5. Freeze interfaces until next checkpoint.

## Submission QA Checklist
1. File count matches expected test count.
2. Filenames match required mapping.
3. All outputs decode and dimensions are correct.
4. Safety metrics are within thresholds.
5. Manifest includes checkpoint, config, and commit hash.

## Required Logging
For each experiment/submission:
- run ID
- commit hash
- config path and overrides
- checkpoint path
- proxy sub-scores + total
- safety and fallback stats
- reviewer notes