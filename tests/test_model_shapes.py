from __future__ import annotations

import torch

from src.models.coord_channels import CoordChannelAppender
from src.models.hybrid_model import HybridLensCorrectionModel, HybridModelConfig


def test_coord_channel_appender_shapes_and_ranges() -> None:
    image = torch.rand(2, 3, 32, 48, dtype=torch.float32)
    app = CoordChannelAppender()
    out = app(image)

    assert out.shape == (2, 6, 32, 48)
    x = out[:, 3]
    y = out[:, 4]
    r = out[:, 5]

    assert x.min().item() >= -1.0001 and x.max().item() <= 1.0001
    assert y.min().item() >= -1.0001 and y.max().item() <= 1.0001
    assert r.min().item() >= 0.0


def test_hybrid_model_output_shapes_bounds_and_init_behavior() -> None:
    cfg = HybridModelConfig(
        backbone_name="tiny",
        use_coord_channels=True,
        include_aspect=False,
        residual_max_disp=4.0,
    )
    model = HybridLensCorrectionModel(config=cfg)

    image = torch.rand(2, 3, 64, 96, dtype=torch.float32)
    out = model(image)

    assert "params" in out
    assert "residual_flow_lowres" in out
    assert "residual_flow_fullres" in out
    assert "debug_stats" in out

    params = out["params"]
    residual_low = out["residual_flow_lowres"]
    residual_full = out["residual_flow_fullres"]

    assert params.shape == (2, 8)
    assert residual_low.shape[0] == 2 and residual_low.shape[1] == 2
    assert residual_full.shape == (2, 2, 64, 96)

    # Bounds checks (default bounds from ParametricBounds).
    assert (params[:, 0] >= -0.6).all() and (params[:, 0] <= 0.6).all()
    assert (params[:, 1] >= -0.3).all() and (params[:, 1] <= 0.3).all()
    assert (params[:, 2] >= -0.15).all() and (params[:, 2] <= 0.15).all()
    assert (params[:, 3] >= -0.03).all() and (params[:, 3] <= 0.03).all()
    assert (params[:, 4] >= -0.03).all() and (params[:, 4] <= 0.03).all()
    assert (params[:, 5] >= -0.08).all() and (params[:, 5] <= 0.08).all()
    assert (params[:, 6] >= -0.08).all() and (params[:, 6] <= 0.08).all()
    assert (params[:, 7] >= 0.90).all() and (params[:, 7] <= 1.20).all()

    # Residual should be bounded and near zero at initialization (zero-init final conv).
    assert residual_low.abs().max().item() <= 4.0001
    assert residual_low.abs().mean().item() < 1e-5