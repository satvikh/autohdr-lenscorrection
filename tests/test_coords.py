import torch

from src.geometry.coords import (
    grid_to_normalized_image,
    grid_to_pixel,
    make_identity_grid,
    normalized_image_to_grid,
    normalized_image_to_pixel,
    pixel_to_grid,
    pixel_to_normalized_image,
    roundtrip_grid_error,
    roundtrip_pixel_error,
)


def test_pixel_normalized_roundtrip_multiple_sizes_and_batched():
    torch.manual_seed(123)
    sizes = [(2, 2), (7, 11), (1, 5), (8, 1), (17, 19)]

    for height, width in sizes:
        batch = 4
        samples = 32

        u = torch.rand(batch, samples, dtype=torch.float64)
        v = torch.rand(batch, samples, dtype=torch.float64)

        u = u * (width - 1) if width > 1 else torch.zeros_like(u)
        v = v * (height - 1) if height > 1 else torch.zeros_like(v)

        x, y = pixel_to_normalized_image(u, v, height, width)
        u_rt, v_rt = normalized_image_to_pixel(x, y, height, width)

        assert torch.allclose(u, u_rt, atol=1e-10, rtol=0.0)
        assert torch.allclose(v, v_rt, atol=1e-10, rtol=0.0)

        du, dv = roundtrip_pixel_error(u, v, height, width)
        assert du.abs().max().item() < 1e-10
        assert dv.abs().max().item() < 1e-10


def test_normalized_grid_pack_unpack_roundtrip():
    torch.manual_seed(7)
    x = (torch.rand(3, 5, 7, dtype=torch.float64) * 2.0) - 1.0
    y = (torch.rand(3, 5, 7, dtype=torch.float64) * 2.0) - 1.0

    grid = normalized_image_to_grid(x, y)
    x_rt, y_rt = grid_to_normalized_image(grid)

    assert grid.shape == (3, 5, 7, 2)
    assert torch.allclose(x, x_rt, atol=0.0, rtol=0.0)
    assert torch.allclose(y, y_rt, atol=0.0, rtol=0.0)


def test_pixel_grid_roundtrip_multiple_sizes():
    torch.manual_seed(99)
    sizes = [(3, 4), (9, 13), (1, 6), (6, 1)]

    for height, width in sizes:
        batch = 2
        n = 40
        u = torch.rand(batch, n, dtype=torch.float64)
        v = torch.rand(batch, n, dtype=torch.float64)

        u = u * (width - 1) if width > 1 else torch.zeros_like(u)
        v = v * (height - 1) if height > 1 else torch.zeros_like(v)

        grid = pixel_to_grid(u, v, height, width)
        u_rt, v_rt = grid_to_pixel(grid, height, width)

        assert grid.shape == (batch, n, 2)
        assert torch.allclose(u, u_rt, atol=1e-10, rtol=0.0)
        assert torch.allclose(v, v_rt, atol=1e-10, rtol=0.0)


def test_grid_roundtrip_helper_on_identity_grid():
    sizes = [(2, 2), (8, 5), (1, 6), (6, 1)]
    for height, width in sizes:
        grid = make_identity_grid(batch=3, height=height, width=width, dtype=torch.float64)
        err = roundtrip_grid_error(grid, height, width)
        assert err.abs().max().item() < 1e-10
