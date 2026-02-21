# Folder convention

## Expected tree
project root
data
val_corrected
pred_val
test
pred_test_param
pred_test_hybrid

tools
proxy_score.py
qa_check.py
select_best.py
build_zip.py

outputs
proxy
selected
submissions

docs
ROLE_PERSON_C.md
METRIC_SPEC.md
PIPELINE_SPEC.md
FOLDER_CONVENTION.md
CHECKLIST.md

## Naming rules
Validation filenames must match between val_corrected and pred_val
Test prediction filenames must match the filenames in data test
File extensions should match across all folders if possible
If not possible the scripts must map by stem name only

## Image rules
All images must be the same width and height for a given id between prediction and target
Color channels must be consistent RGB
No alpha channel unless the competition explicitly allows it