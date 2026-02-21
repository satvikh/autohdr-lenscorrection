from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference.predictor import Predictor
from src.inference.writer import save_jpeg


class StubNeutralModel:
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        params = torch.zeros((b, 8), dtype=image.dtype, device=image.device)
        params[:, 7] = 1.0
        return {"params": params}


def _make_test_image(path: Path, h: int = 32, w: int = 40) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.int32),
        np.arange(w, dtype=np.int32),
        indexing="ij",
    )
    r = ((xx * 5 + yy * 3) % 256).astype(np.uint8)
    g = ((xx * 7 + yy * 11) % 256).astype(np.uint8)
    b = ((xx * 13 + yy * 17) % 256).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)

    Image.fromarray(img, mode="RGB").save(path)
    return img


def test_param_only_inference_identity_and_jpeg_roundtrip(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.jpg"

    _make_test_image(input_path)

    predictor = Predictor(model=StubNeutralModel())
    warped, metadata = predictor.predict(input_path)

    assert metadata["mode_used"] == "param_only"
    assert metadata["safe"] is True
    assert tuple(warped.shape) == (1, 3, 32, 40)

    # Identity params should reproduce input exactly up to interpolation tolerance.
    with Image.open(input_path) as img:
        in_arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    in_tensor = torch.from_numpy(in_arr).permute(2, 0, 1).unsqueeze(0)

    max_err = (warped - in_tensor).abs().max().item()
    assert max_err < 1e-5, f"max_err={max_err}"

    saved_h, saved_w = save_jpeg(warped, output_path, expected_hw=(32, 40))
    assert (saved_h, saved_w) == (32, 40)
    assert output_path.exists()

    with Image.open(output_path) as out_img:
        out_rgb = out_img.convert("RGB")
        assert out_rgb.size == (40, 32)
