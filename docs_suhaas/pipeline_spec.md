# Pipeline specification

## Inputs
data val_corrected
Ground truth corrected validation images

data pred_val
Predicted corrected validation images

data test
Test images used to determine required filenames

data pred_test_param
Candidate test predictions from parametric only model

data pred_test_hybrid
Candidate test predictions from hybrid model

## Outputs
outputs proxy proxy_report.csv
Per image proxy scores and component scores

outputs proxy fail_report.csv
Per image failure flags and reasons

outputs selected
Folder containing the selected best safe test images

outputs submissions submission.zip
Zip file containing final selected test outputs

outputs submissions manifest.json
Metadata about the submission zip including run name checkpoint and counts

outputs experiment_log.csv
One row per experiment and submission attempt

## Execution order
1 Run proxy scoring on validation predictions
2 Run QA checks on validation predictions and confirm no obvious border artifacts
3 If multiple candidate folders exist run selection to choose best per image
4 Run QA checks on selected test outputs
5 Build the submission zip and manifest
6 Append experiment log row with proxy results and notes

## Selection policy
Prefer the hybrid output if it passes safety checks
Fallback to parametric output if hybrid fails
If both fail pick the one with lower border artifacts and lower proxy worst patch error

## Key constraints
Outputs must preserve original resolution
Outputs must not introduce black borders or empty corners
Outputs must be valid image files that decode correctly
Zip must contain exactly one image per test id with correct filenames