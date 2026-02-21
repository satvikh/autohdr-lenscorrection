# Role Person C

## What I own
Proxy scoring that approximates the official metric
Quality assurance checks that prevent hard fail outputs
Per image safety gates and fallback selection between multiple prediction variants
Submission packaging into a zip in the required format
Experiment logging and reporting so the team knows what improved score

## What I do not own
Model training code and training loops
Warp math and coordinate conversions
Loss design and tuning
Neural network architecture decisions
GPU environment setup

## Goal
Produce a safe high scoring submission zip with zero catastrophic images
Enable fast iteration by ranking model outputs locally with a proxy score
Prevent score loss by catching border artifacts and local stretch failures

## What I need from teammates
Validation ground truth folder with corrected images
Validation predictions folder with identical filenames
Test predictions folder or folders with identical filenames to test set
Preferred output format and exact expected zip structure from the competition rules

## Required folder convention
data
val_corrected
pred_val
test
pred_test_param
pred_test_hybrid

Filenames must match exactly across val_corrected and pred_val
Filenames must match exactly across test and prediction folders

## My scripts
tools proxy_score.py
Computes proxy metrics and aggregates weighted score

tools qa_check.py
Checks image integrity dimensions borders and suspicious artifacts

tools select_best.py
Selects the best safe output per image from multiple candidate folders

tools build_zip.py
Creates submission zip and manifest

## Decision policy for selecting outputs
If hybrid output passes safety checks use hybrid
Else use parametric output
If both fail mark the image and still pick the one with the least border artifacts and lowest worst patch error on proxy checks

## Definition of done
Proxy scorer runs on validation set end to end and produces a report csv
QA script passes on selected test outputs
Zip builder produces a zip with correct file count and filenames
We have a manifest json for every submission and a log row in experiment_log.csv