# Role Person C (Person 3)

## Role Summary
Person 3 owns Data + Proxy Scorer + QA + Submission Tooling.

## What I own
- Proxy scoring that approximates the official metric.
- Quality assurance checks that prevent hard-fail outputs.
- Submission packaging and manifest generation.
- Experiment logging and comparative reporting.

## What I do not own
- Model training internals.
- Warp math and coordinate conversion internals.
- Loss design and tuning internals.

## Core Deliverables
- data audit and split scripts
- data loaders and transforms
- proxy metrics and aggregate scorer
- QA scripts for filenames and image integrity
- submission zip builder and manifest writer
- experiment logs for ablations and submissions

## Decision Policy for Output Selection
1. Prefer hybrid output when safety checks pass.
2. Fallback to parametric output if hybrid fails.
3. If both fail, select least risky candidate by border/error heuristics and flag.

## Definition of Done
- Proxy scorer runs end-to-end on validation data.
- QA scripts catch malformed outputs.
- Submission zip and manifest are reproducible.
- Experiment/submission logs are complete and traceable.