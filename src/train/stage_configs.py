from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageToggles:
    """Stage-level behavior toggles for training.

    Attributes:
        name: canonical stage name.
        use_residual: whether residual branch output is consumed.
        use_flow_regularizers: whether flow TV/magnitude/curvature losses are enabled.
        use_jacobian_penalty: whether jacobian foldover penalty is enabled.
    """

    name: str
    use_residual: bool
    use_flow_regularizers: bool
    use_jacobian_penalty: bool


def get_stage_toggles(stage: str) -> StageToggles:
    s = stage.lower()
    if s == "stage1_param_only":
        return StageToggles(
            name=s,
            use_residual=False,
            use_flow_regularizers=False,
            use_jacobian_penalty=False,
        )
    if s == "stage2_hybrid":
        return StageToggles(
            name=s,
            use_residual=True,
            use_flow_regularizers=True,
            use_jacobian_penalty=True,
        )
    if s == "stage3_finetune":
        return StageToggles(
            name=s,
            use_residual=True,
            use_flow_regularizers=True,
            use_jacobian_penalty=True,
        )
    raise ValueError(f"Unsupported stage: {stage}")