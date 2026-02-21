import torch

from src.geometry.coords import make_identity_grid
from src.geometry.warp_ops import warp_image


def test_identity_warp_reconstructs_input():
    torch.manual_seed(202)

    batch, channels, height, width = 2, 3, 17, 19

    base = torch.arange(batch * channels * height * width, dtype=torch.float32)
    image = (base.reshape(batch, channels, height, width) % 257) / 256.0

    grid = make_identity_grid(batch=batch, height=height, width=width, dtype=image.dtype)
    warped = warp_image(image, grid, mode="bilinear", padding_mode="border", align_corners=True)

    assert warped.shape == image.shape
    assert torch.allclose(warped, image, atol=1e-6, rtol=0.0)
