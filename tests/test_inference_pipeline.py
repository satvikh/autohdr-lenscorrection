from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor
from src.inference.writer import save_jpeg


class StubNeutralModel:
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        params = torch.zeros((b, 8), dtype=image.dtype, device=image.device)
        params[:, 7] = 1.0
        return {"params": params}


class StubUnsafeThenConservativeSafeModel:
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        params = torch.zeros((b, 8), dtype=image.dtype, device=image.device)
        # Intentionally unsafe in baseline (triggers OOB), but conservative clamp should recover.
        params[:, 0] = -0.21210544
        params[:, 1] = -0.16145031
        params[:, 2] = 0.14677799
        params[:, 3] = -0.01208510
        params[:, 4] = 0.00360298
        params[:, 5] = 0.02625185
        params[:, 6] = -0.07400852
        params[:, 7] = 0.92668712
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
    assert metadata["safety"]["safe"] is True
    assert tuple(warped.shape) == (1, 3, 32, 40)

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


def test_fallback_to_conservative_and_safe_final(tmp_path: Path):
    input_path = tmp_path / "input2.png"
    _make_test_image(input_path)

    predictor = Predictor(model=StubUnsafeThenConservativeSafeModel())
    warped, metadata = predictor.predict(input_path)

    assert metadata["mode_used"] == "param_only_conservative"
    assert isinstance(metadata["warnings"], list)
    assert len(metadata["warnings"]) > 0
    assert metadata["safety"]["safe"] is True
    assert metadata["jacobian"]["negative_det_pct"] >= 0.0
    assert tuple(warped.shape) == (1, 3, 32, 40)


def _run_all_with_tmpdir() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_param_only_inference_identity_and_jpeg_roundtrip(tmp_path)
        test_fallback_to_conservative_and_safe_final(tmp_path)


if __name__ == "__main__":
    _run_all_with_tmpdir()
    print("test_inference_pipeline.py: PASS")
