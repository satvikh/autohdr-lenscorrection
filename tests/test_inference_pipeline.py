from __future__ import annotations

import json
from pathlib import Path
import subprocess
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


class StubHybridSmallResidualModel:
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        h = image.shape[2]
        w = image.shape[3]
        params = torch.zeros((b, 8), dtype=image.dtype, device=image.device)
        params[:, 7] = 1.0
        # BHWC pixel residual with dx=1 in interior only (edges stay 0 to avoid OOB).
        residual = torch.zeros((b, h, w, 2), dtype=image.dtype, device=image.device)
        residual[:, 1:-1, 1:-1, 0] = 1.0
        return {"params": params, "residual_flow": residual}


class StubHybridHugeResidualModel:
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        params = torch.zeros((b, 8), dtype=image.dtype, device=image.device)
        params[:, 7] = 1.0
        # BCHW pixel residual, huge magnitude should be unsafe in hybrid.
        residual = torch.full((b, 2, 4, 5), 200.0, dtype=image.dtype, device=image.device)
        return {"params": params, "residual_flow": residual}


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


def test_hybrid_small_residual_safe(tmp_path: Path):
    input_path = tmp_path / "input_hybrid_safe.png"
    _make_test_image(input_path)

    predictor = Predictor(model=StubHybridSmallResidualModel())
    _, metadata = predictor.predict(input_path)

    assert metadata["mode_used"] == "hybrid"
    assert metadata["safety"]["safe"] is True
    assert isinstance(metadata["warnings"], list)


def test_hybrid_huge_residual_fallback(tmp_path: Path):
    input_path = tmp_path / "input_hybrid_unsafe.png"
    _make_test_image(input_path)

    predictor = Predictor(model=StubHybridHugeResidualModel())
    _, metadata = predictor.predict(input_path)

    assert metadata["mode_used"] in ("param_only", "param_only_conservative")
    assert isinstance(metadata["warnings"], list)
    assert len(metadata["warnings"]) > 0
    assert metadata["safety"]["safe"] is True


def test_infer_test_writes_run_metadata_json(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _make_test_image(input_dir / "img_a.png", h=16, w=20)
    _make_test_image(input_dir / "img_b.png", h=18, w=22)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("resize_to: null\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer_test.py"),
        str(input_dir),
        str(output_dir),
        "--checkpoint-id",
        "ckpt_test_001",
        "--config-path",
        str(config_path),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    metadata_path = output_dir / "run_metadata.json"
    assert metadata_path.exists()

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["checkpoint_id"] == "ckpt_test_001"
    assert data["config_hash"] != "none"
    assert data["processed"] == 2
    assert isinstance(data["mode_counts"], dict)
    assert "timestamp_utc" in data


def _run_all_with_tmpdir() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_param_only_inference_identity_and_jpeg_roundtrip(tmp_path)
        test_hybrid_small_residual_safe(tmp_path)
        test_hybrid_huge_residual_fallback(tmp_path)
        test_infer_test_writes_run_metadata_json(tmp_path)


if __name__ == "__main__":
    _run_all_with_tmpdir()
    print("test_inference_pipeline.py: PASS")
