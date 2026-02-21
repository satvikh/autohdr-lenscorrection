from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.coords import make_identity_grid
from src.inference.fallback import run_fallback_hierarchy
from src.inference.safety import SafetyConfig, evaluate_safety


def test_unsafe_grid_reports_not_safe_and_reasons():
    grid = make_identity_grid(batch=1, height=17, width=17, dtype=torch.float32)

    # Force out-of-bounds and foldover simultaneously.
    grid[..., 0] = -1.5 * grid[..., 0]

    cfg = SafetyConfig(
        max_out_of_bounds_ratio=0.0,
        max_negative_det_pct=0.0,
        min_det_min=0.0,
        min_det_p01=0.0,
    )
    report = evaluate_safety(grid, config=cfg)

    assert report["safe"] is False
    assert len(report["reasons"]) > 0
    assert report["metrics"]["out_of_bounds_ratio"] > 0.0
    assert report["metrics"]["negative_det_pct"] > 0.0


def test_fallback_order_and_hard_unsafe_warning():
    calls: list[str] = []

    def always_unsafe(params: torch.Tensor, mode: str):
        calls.append(mode)
        return {"safe": False, "reasons": [f"{mode}_unsafe"], "metrics": {}}

    hybrid = torch.zeros((1, 8), dtype=torch.float32)
    hybrid[:, 7] = 1.0
    param = torch.zeros((1, 8), dtype=torch.float32)
    param[:, 7] = 1.0

    mode_used, chosen_params, warnings, safety_report = run_fallback_hierarchy(
        hybrid,
        param,
        always_unsafe,
    )

    assert calls == ["hybrid", "param_only", "param_only_conservative"]
    assert mode_used == "param_only_conservative"
    assert chosen_params.shape == (1, 8)
    assert warnings[-1] == "HARD_UNSAFE_OUTPUT"
    assert safety_report["safe"] is False


def _run_all():
    test_unsafe_grid_reports_not_safe_and_reasons()
    test_fallback_order_and_hard_unsafe_warning()


if __name__ == "__main__":
    _run_all()
    print("test_safety_fallback.py: PASS")
