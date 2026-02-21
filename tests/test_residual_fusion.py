from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.residual_fusion import (
    adapt_and_normalize_residual_flow,
    adapt_residual_flow_to_bhwc,
    pixel_flow_to_normalized_delta,
    upsample_residual_flow,
)


def test_adapter_accepts_bchw_and_bhwc():
    bchw = torch.tensor(
        [[
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        ]]
    )  # [1,2,3,2]
    bhwc = torch.tensor(
        [[
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ]]
    )  # [1,3,2,2]

    out_from_bchw = adapt_residual_flow_to_bhwc(bchw)
    out_from_bhwc = adapt_residual_flow_to_bhwc(bhwc)

    assert out_from_bchw.shape == (1, 3, 2, 2)
    assert out_from_bhwc.shape == (1, 3, 2, 2)
    assert torch.allclose(out_from_bchw, bhwc)
    assert torch.allclose(out_from_bhwc, bhwc)


def test_pixel_to_normalized_matches_formula_tiny_tensor():
    # Hr=3, Wr=5 => dx_norm = dx_px * 2/(5-1) = dx_px * 0.5
    #               dy_norm = dy_px * 2/(3-1) = dy_px * 1.0
    flow_px = torch.tensor(
        [[[[2.0, -1.0], [4.0, 3.0], [0.0, 2.0], [1.0, -2.0], [6.0, 5.0]],
          [[-2.0, 7.0], [3.0, -3.0], [2.0, 1.0], [0.0, 0.0], [8.0, -4.0]],
          [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]]],
        dtype=torch.float32,
    )

    flow_norm = pixel_flow_to_normalized_delta(flow_px)

    expected_dx = flow_px[..., 0] * 0.5
    expected_dy = flow_px[..., 1] * 1.0

    assert torch.allclose(flow_norm[..., 0], expected_dx)
    assert torch.allclose(flow_norm[..., 1], expected_dy)


def test_upsample_shape_and_finite():
    flow_bchw_px = torch.randn(2, 2, 4, 6, dtype=torch.float32)
    flow_norm_bhwc = adapt_and_normalize_residual_flow(flow_bchw_px)

    up = upsample_residual_flow(flow_norm_bhwc, target_h=9, target_w=11, align_corners=True)

    assert up.shape == (2, 9, 11, 2)
    assert torch.isfinite(up).all()


def _run_all():
    test_adapter_accepts_bchw_and_bhwc()
    test_pixel_to_normalized_matches_formula_tiny_tensor()
    test_upsample_shape_and_finite()


if __name__ == "__main__":
    _run_all()
    print("test_residual_fusion.py: PASS")
