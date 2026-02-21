from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _to_tensor(value: Tensor | float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def pixel_to_normalized_image(
    u: Tensor | float,
    v: Tensor | float,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[Tensor, Tensor]:
    """Convert pixel coordinates (u, v) to normalized image coordinates (x, y).

    This uses the align_corners=True convention.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    if isinstance(u, Tensor):
        device = u.device if device is None else device
        dtype = u.dtype if dtype is None else dtype
    elif isinstance(v, Tensor):
        device = v.device if device is None else device
        dtype = v.dtype if dtype is None else dtype
    else:
        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

    u_t = _to_tensor(u, device=device, dtype=dtype)
    v_t = _to_tensor(v, device=device, dtype=dtype)

    if width > 1:
        x = (2.0 * u_t / (width - 1)) - 1.0
    else:
        x = torch.zeros_like(u_t)

    if height > 1:
        y = (2.0 * v_t / (height - 1)) - 1.0
    else:
        y = torch.zeros_like(v_t)

    return x, y


def normalized_image_to_pixel(
    x: Tensor | float,
    y: Tensor | float,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[Tensor, Tensor]:
    """Convert normalized image coordinates (x, y) to pixel coordinates (u, v).

    This uses the align_corners=True convention.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    if isinstance(x, Tensor):
        device = x.device if device is None else device
        dtype = x.dtype if dtype is None else dtype
    elif isinstance(y, Tensor):
        device = y.device if device is None else device
        dtype = y.dtype if dtype is None else dtype
    else:
        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

    x_t = _to_tensor(x, device=device, dtype=dtype)
    y_t = _to_tensor(y, device=device, dtype=dtype)

    if width > 1:
        u = ((x_t + 1.0) * (width - 1)) / 2.0
    else:
        u = torch.zeros_like(x_t)

    if height > 1:
        v = ((y_t + 1.0) * (height - 1)) / 2.0
    else:
        v = torch.zeros_like(y_t)

    return u, v


def normalized_image_to_grid(x: Tensor, y: Tensor) -> Tensor:
    """Pack normalized image coordinates into BHWC grid layout expected by grid_sample."""
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    return torch.stack((x, y), dim=-1)


def grid_to_normalized_image(grid: Tensor) -> Tuple[Tensor, Tensor]:
    """Unpack BHWC grid into normalized image coordinates (x, y)."""
    if grid.ndim < 1 or grid.shape[-1] != 2:
        raise ValueError("grid must have last dimension of size 2")
    return grid[..., 0], grid[..., 1]


def pixel_to_grid(
    u: Tensor,
    v: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Convert pixel coordinates (u, v) to normalized BHWC grid coordinates."""
    x, y = pixel_to_normalized_image(u, v, height, width)
    return normalized_image_to_grid(x, y)


def grid_to_pixel(
    grid: Tensor,
    height: int,
    width: int,
) -> Tuple[Tensor, Tensor]:
    """Convert normalized BHWC grid coordinates to pixel coordinates (u, v)."""
    x, y = grid_to_normalized_image(grid)
    return normalized_image_to_pixel(x, y, height, width)


def make_identity_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Build an identity backward-warp grid in BHWC format using align_corners=True."""
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError("batch, height, and width must be positive")

    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype

    if width > 1:
        gx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    else:
        gx = torch.zeros(1, device=device, dtype=dtype)

    if height > 1:
        gy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    else:
        gy = torch.zeros(1, device=device, dtype=dtype)

    y_grid, x_grid = torch.meshgrid(gy, gx, indexing="ij")
    base = torch.stack((x_grid, y_grid), dim=-1)
    return base.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def roundtrip_pixel_error(u: Tensor, v: Tensor, height: int, width: int) -> Tuple[Tensor, Tensor]:
    """Round-trip helper: pixel -> normalized -> pixel error."""
    x, y = pixel_to_normalized_image(u, v, height, width)
    u_rt, v_rt = normalized_image_to_pixel(x, y, height, width)
    return u_rt - u, v_rt - v


def roundtrip_grid_error(grid: Tensor, height: int, width: int) -> Tensor:
    """Round-trip helper: grid -> pixel -> grid error."""
    u, v = grid_to_pixel(grid, height, width)
    grid_rt = pixel_to_grid(u, v, height, width)
    return grid_rt - grid
