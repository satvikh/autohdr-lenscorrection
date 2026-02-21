# Metric Specification (Person 3)

## Score Summary
Local proxy score should approximate geometry-focused official scoring behavior.

## Component Weights
- Edge similarity: 0.40
- Line straightness: 0.22
- Gradient orientation similarity: 0.18
- SSIM: 0.15
- Pixel accuracy (MAE-derived): 0.05

## Components
### Edge Similarity
- Multi-scale Canny-style edge comparison between prediction and target.

### Line Straightness
- Compare long-line orientation and straightness distributions (Hough/LSD style).

### Gradient Orientation Similarity
- Compare gradient direction histograms, weighted by gradient magnitude.

### SSIM
- Structural similarity between prediction and target.

### Pixel Accuracy
- MAE-based component with low relative weight.

## Hard-Fail Considerations
Some competition settings penalize catastrophic outputs heavily (including near-zero scoring for severe failures).

Proxy hard-fail checks should flag:
- catastrophic local distortions
- severe border artifacts
- extreme edge mismatch or collapse

## Calibration Guidance
- GT-vs-GT proxy should be near perfect.
- Distorted-vs-GT proxy should be meaningfully worse.
- Periodically calibrate proxy trends against external submissions.