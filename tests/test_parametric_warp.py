import torch

from src.geometry.coords import make_identity_grid
from src.geometry.parametric_warp import (
    DC_RANGE,
    K1_RANGE,
    K2_RANGE,
    K3_RANGE,
    P_RANGE,
    S_RANGE,
    build_parametric_grid,
)


def _neutral_params(batch: int, dtype: torch.dtype) -> torch.Tensor:
    params = torch.zeros((batch, 8), dtype=dtype)
    params[:, 7] = 1.0  # s
    return params


def test_neutral_params_produce_identity_grid():
    sizes = [(2, 2), (9, 13), (17, 19)]

    for dtype, tol in ((torch.float32, 1e-6), (torch.float64, 1e-10)):
        for height, width in sizes:
            params = _neutral_params(batch=3, dtype=dtype)
            grid = build_parametric_grid(
                params,
                height,
                width,
                align_corners=True,
                device=torch.device("cpu"),
                dtype=dtype,
            )
            identity = make_identity_grid(3, height, width, device=torch.device("cpu"), dtype=dtype)

            err = (grid - identity).abs().max().item()
            assert err < tol


def test_random_in_range_has_no_nans_or_infs():
    torch.manual_seed(11)
    batch = 32
    params = torch.empty((batch, 8), dtype=torch.float32)

    params[:, 0].uniform_(*K1_RANGE)
    params[:, 1].uniform_(*K2_RANGE)
    params[:, 2].uniform_(*K3_RANGE)
    params[:, 3].uniform_(*P_RANGE)
    params[:, 4].uniform_(*P_RANGE)
    params[:, 5].uniform_(*DC_RANGE)
    params[:, 6].uniform_(*DC_RANGE)
    params[:, 7].uniform_(*S_RANGE)

    grid = build_parametric_grid(
        params,
        height=31,
        width=37,
        align_corners=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.isfinite(grid).all()


def test_positive_k1_radial_sanity_at_corners():
    height, width = 33, 33

    neutral = _neutral_params(batch=1, dtype=torch.float64)
    positive = _neutral_params(batch=1, dtype=torch.float64)
    negative = _neutral_params(batch=1, dtype=torch.float64)

    positive[:, 0] = 0.25  # k1
    negative[:, 0] = -0.25  # k1

    g_neutral = build_parametric_grid(neutral, height, width, True, torch.device("cpu"), torch.float64)
    g_positive = build_parametric_grid(positive, height, width, True, torch.device("cpu"), torch.float64)
    g_negative = build_parametric_grid(negative, height, width, True, torch.device("cpu"), torch.float64)

    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]

    for i, j in corners:
        rn = torch.linalg.vector_norm(g_neutral[0, i, j]).item()
        rp = torch.linalg.vector_norm(g_positive[0, i, j]).item()
        rm = torch.linalg.vector_norm(g_negative[0, i, j]).item()

        # Coarse monotonic radial check at the image corners.
        assert rp > rn
        assert rm < rn
