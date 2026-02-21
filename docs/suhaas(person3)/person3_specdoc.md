# Person 3 Spec Doc: Data + Proxy + QA + Submission Lead

## Purpose
Define Person 3 responsibilities and interfaces for data readiness, local scoring, and submission safety.

## Primary Objective
Provide reliable data pipeline and scoring/QA systems so model improvements can be measured quickly and shipped safely.

## Out of Scope
- Geometry warp internals and inference fallback implementation (Person 1).
- Model architecture and training internals (Person 2).

## Required Contracts
Person 3 owns and must enforce:
- dataset sample schema
- proxy scorer return schema
- QA/submission artifact schema

## Data Responsibilities
- dataset audit reporting
- split generation and reproducibility
- paired transforms that preserve geometric mapping

## Proxy Responsibilities
- edge/line/gradient/ssim/mae sub-scores
- weighted aggregate score
- hard-fail flags and reasons
- validation scripts for fast iteration

## QA Responsibilities
- filename checks
- image decode/integrity checks
- dimension consistency checks
- submission manifest generation

## Acceptance Criteria
1. Audit + splits generated and versioned.
2. Proxy scorer stable and fast.
3. QA scripts catch common submission failures.
4. Submission zip + manifest pipeline is deterministic.

## Risks and Mitigations
Risk: proxy misalignment with external metric.
- Mitigation: periodic calibration against external submissions.

Risk: filename or packaging mistakes.
- Mitigation: mandatory pre-zip QA checks + manifest validation.

Risk: data transforms breaking paired geometry.
- Mitigation: restrict augmentations to geometry-preserving operations.

## Deliverables
- data modules/scripts
- metrics modules/scripts
- QA modules/scripts
- experiment and submission logs