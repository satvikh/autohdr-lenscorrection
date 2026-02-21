from __future__ import annotations

import torch

from src.geometry.coords import make_identity_grid
from src.losses.composite import CompositeLoss, CompositeLossConfig, config_for_stage
from src.losses.flow_regularizers import flow_curvature_loss, flow_magnitude_loss, total_variation_loss
from src.losses.gradients import edge_magnitude_loss, gradient_orientation_cosine_loss
from src.losses.jacobian_loss import jacobian_foldover_penalty
from src.losses.pixel import CharbonnierLoss, l1_loss
from src.losses.ssim_loss import SSIMLoss


def test_basic_losses_identical_inputs_are_small() -> None:
    pred = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    target = pred.clone()

    assert l1_loss(pred, target).item() < 1e-7
    assert SSIMLoss()(pred, target).item() < 1e-5
    assert edge_magnitude_loss(pred, target).item() < 1e-6
    assert gradient_orientation_cosine_loss(pred, target).item() < 5e-6

    charbonnier = CharbonnierLoss(eps=1e-3)
    # Charbonnier has non-zero floor around eps.
    assert abs(charbonnier(pred, target).item() - 1e-3) < 3e-4


def test_losses_increase_with_structural_mismatch() -> None:
    target = torch.zeros(1, 3, 32, 32)
    pred = target.clone()
    pred[:, :, 8:24, 8:24] = 1.0

    assert l1_loss(pred, target).item() > 0.0
    assert edge_magnitude_loss(pred, target).item() > 0.0
    assert gradient_orientation_cosine_loss(pred, target).item() > 0.0


def test_flow_regularizers_and_jacobian_penalty() -> None:
    flow = torch.randn(2, 2, 16, 16) * 0.1
    assert total_variation_loss(flow).item() >= 0.0
    assert flow_magnitude_loss(flow).item() >= 0.0
    assert flow_curvature_loss(flow).item() >= 0.0

    identity = make_identity_grid(batch=1, height=21, width=21, dtype=torch.float32)
    flipped = identity.clone()
    flipped[..., 0] = -flipped[..., 0]

    pen_id = jacobian_foldover_penalty(identity)
    pen_flip = jacobian_foldover_penalty(flipped)

    assert pen_id.item() < 1e-6
    assert pen_flip.item() > 0.0


def test_composite_loss_stage_behavior() -> None:
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    residual = torch.randn(2, 2, 8, 8) * 0.01
    grid = make_identity_grid(batch=2, height=64, width=64, dtype=torch.float32)

    cfg1 = config_for_stage("stage1_param_only")
    loss1 = CompositeLoss(cfg1)
    total1, comps1 = loss1(pred, target, residual_flow_lowres=None, final_grid_bhwc=None)

    assert torch.isfinite(total1)
    assert "total" in comps1
    assert comps1["flow_tv"].item() == 0.0
    assert comps1["jacobian"].item() == 0.0

    cfg2 = CompositeLossConfig(
        stage="stage2_hybrid",
        use_charbonnier=False,
        multiscale_scales=(1.0,),
        pixel_weight=0.1,
        ssim_weight=0.1,
        edge_weight=0.1,
        grad_orient_weight=0.1,
        flow_tv_weight=0.1,
        flow_mag_weight=0.1,
        flow_curv_weight=0.1,
        jacobian_weight=0.1,
    )
    loss2 = CompositeLoss(cfg2)
    total2, comps2 = loss2(pred, target, residual_flow_lowres=residual, final_grid_bhwc=grid)

    assert torch.isfinite(total2)
    assert comps2["flow_tv"].item() >= 0.0
    assert comps2["flow_mag"].item() >= 0.0
    assert comps2["jacobian"].item() >= 0.0
