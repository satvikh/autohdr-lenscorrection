from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.data.dataset import PairedLensDataset
from src.data.transforms import PairedTransforms


def _write_rgb_image(path, hw, value):
    h, w = hw
    array = np.full((h, w, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_paired_dataset_and_transforms(tmp_path):
    input_path = tmp_path / "a_input.png"
    target_path = tmp_path / "a_target.png"
    _write_rgb_image(input_path, (64, 96), 50)
    _write_rgb_image(target_path, (64, 96), 200)

    split_csv = tmp_path / "train_split.csv"
    with split_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "input_path", "target_path"])
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "sample_a",
                "input_path": str(input_path),
                "target_path": str(target_path),
            }
        )

    cfg = SimpleNamespace(
        resize_hw=(32, 48),
        center_crop_hw=None,
        random_crop_hw=None,
        hflip_prob=0.0,
        seed=7,
    )
    transforms = PairedTransforms(cfg)
    dataset = PairedLensDataset(split_csv=split_csv, transforms=transforms, return_paths=True)

    sample = dataset[0]

    assert "input_image" in sample
    assert "target_image" in sample
    assert "image_id" in sample
    assert "orig_size" in sample
    assert "input_path" in sample
    assert "target_path" in sample

    x = sample["input_image"]
    y = sample["target_image"]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert x.shape == (3, 32, 48)
    assert y.shape == (3, 32, 48)
    assert sample["orig_size"] == (64, 96)
    assert sample["image_id"] == "sample_a"
    assert sample["input_path"] == str(input_path)
    assert sample["target_path"] == str(target_path)
