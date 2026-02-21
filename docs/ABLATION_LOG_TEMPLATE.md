# Ablation Log Template

Use one block per experiment run.

## Run Metadata
- Run ID:
- Date:
- Code revision:
- Checkpoint base:
- Config file:
- Data split:

## Change Under Test
- Primary change:
- Secondary changes:
- Why this change should help:

## Key Settings
- Input resolution:
- Interpolation mode:
- Residual max displacement:
- Param bounds adjustments:
- Loss weights summary:
- Safety thresholds summary:

## Results
- Local proxy aggregate:
- Edge component:
- Line component:
- Gradient orientation component:
- SSIM component:
- Pixel component:
- Safety failure count/rate:
- Fallback usage counts:

## Visual Review
- Typical improvements observed:
- Typical regressions observed:
- Border/corner issues:
- Scene types that got worse:

## External Submission (if performed)
- Submission ID/version:
- Leaderboard score:
- Delta vs previous submission:

## Decision
- Keep change: yes/no
- If no, rollback reason:
- Next experiment: