from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_rgb_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32) / 255.0
    return tensor.contiguous()


class PairedLensDataset(Dataset):
    REQUIRED_COLUMNS = {"image_id", "input_path", "target_path"}

    def __init__(
        self,
        split_csv: str | Path,
        transforms: Optional[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
        return_paths: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.transforms = transforms
        self.return_paths = bool(return_paths)

        with self.split_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("Split CSV has no header row.")
            missing = self.REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing required columns in split CSV: {sorted(missing)}")

            self.rows = [
                {
                    "image_id": str(row["image_id"]),
                    "input_path": str(row["input_path"]),
                    "target_path": str(row["target_path"]),
                }
                for row in reader
            ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        input_path = row["input_path"]
        target_path = row["target_path"]
        image_id = row["image_id"]

        input_image = _load_rgb_tensor(input_path)
        target_image = _load_rgb_tensor(target_path)

        h0, w0 = int(input_image.shape[-2]), int(input_image.shape[-1])

        if self.transforms is not None:
            input_image, target_image = self.transforms(input_image, target_image)

        sample: Dict[str, object] = {
            "input_image": input_image,
            "target_image": target_image,
            "image_id": image_id,
            "orig_size": (h0, w0),
        }
        if self.return_paths:
            sample["input_path"] = input_path
            sample["target_path"] = target_path
        return sample
