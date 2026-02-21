# Metric specification

## Official score summary
Per image score combines several similarity measures focused on geometry and structure
Final leaderboard metric is mean absolute error from the perfect score 1.0 over all test images

## Component weights
Edge similarity weight 0.40
Line straightness weight 0.22
Gradient orientation similarity weight 0.18
SSIM weight 0.15
Pixel accuracy weight 0.05

## Edge similarity
Edge similarity uses multi scale Canny edge extraction and compares predicted edges to target edges

## Line straightness
Line straightness compares the angle distribution of long lines detected using Hough style line detection

## Gradient orientation similarity
Compares gradient direction histograms weighted by gradient magnitude

## SSIM
Structural similarity index between prediction and target

## Pixel accuracy
Mean absolute error between prediction and target

## Hard fail notes
There are failure conditions that can assign an image score of 0.0
Examples include catastrophic local errors such as extreme regional differences
And edge similarity falling below a minimum threshold
Therefore border artifacts holes and severe local warps must be prevented