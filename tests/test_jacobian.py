from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.coords import make_identity_grid
from src.geometry.jacobian import jacobian_stats


def test_identity_grid_det_near_one():
    grid = make_identity_grid(batch=2, height=17, width=19, dtype=torch.float64)
    stats = jacobian_stats(grid)

    assert abs(stats["det_mean"] - 1.0) < 1e-6
    assert stats["det_min"] > 0.999
    assert stats["negative_det_pct"] == 0.0


def test_synthetic_foldover_has_negative_det_pct():
    # Horizontal flip produces negative determinant (orientation reversal).
    grid = make_identity_grid(batch=1, height=21, width=21, dtype=torch.float32)
    grid[..., 0] = -grid[..., 0]

    stats = jacobian_stats(grid)

    assert stats["negative_det_pct"] > 0.0
    assert stats["det_min"] < 0.0


def _run_all():
    test_identity_grid_det_near_one()
    test_synthetic_foldover_has_negative_det_pct()


if __name__ == "__main__":
    _run_all()
    print("test_jacobian.py: PASS")
