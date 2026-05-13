"""PyTorch Dataset wrapping a catalog DataFrame slice.

Used for both evaluation (no label needed) and training (label = `class_id`,
the (action, uid) global int assigned by :func:`assign_class_ids`).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ReIDImageDataset(Dataset):
    """Image dataset for feature extraction and training.

    Each item returns a dict:
        {
            "image":      torch.Tensor [C, H, W],
            "bbox_idx":   int,
            "action_idx": int,
            "person_uid": int,         # -1 for challenge (no annotations)
            "class_id":   int          # included only if the DataFrame has a
                                       # `class_id` column (training mode);
                                       # -1 for unannotated rows.
        }

    `transform` is a callable that takes a PIL.Image and returns a tensor (see
    :mod:`soccernet_reid.transforms`).
    """

    def __init__(self, df: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._has_class_id = "class_id" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            img = self.transform(im)
        person_uid = row["person_uid"]
        if pd.isna(person_uid):
            person_uid = -1
        out: dict[str, object] = {
            "image": img,
            "bbox_idx": int(row["bbox_idx"]),
            "action_idx": int(row["action_idx"]),
            "person_uid": int(person_uid),
        }
        if self._has_class_id:
            out["class_id"] = int(row["class_id"])
        return out


def default_eval_transform(height: int = 256, width: int = 128):
    """Standard evaluation transform: resize → tensor [0,1] → ImageNet normalize."""
    from torchvision.transforms import v2

    return v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize((height, width), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
