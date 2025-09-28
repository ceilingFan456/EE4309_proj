# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
import deeplake
from PIL import Image

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VOC_CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}  # 0 is background


class VOCDataset(Dataset):
    def __init__(self, image_set="trainval", transforms=None, keep_difficult=False):
        if image_set == "trainval":
            url = "hub://activeloop/pascal-voc-2007-train-val"
        else:
            url = "hub://activeloop/pascal-voc-2007-test"
        self.dataset = deeplake.load(url)
        self.transforms = transforms
        self.keep_difficult = keep_difficult

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        item = self.dataset[idx]
        img = Image.fromarray(item["images"].numpy())

        boxes = item["boxes/box"].numpy()
        labels = item["boxes/label"].numpy().flatten()
        difficult = item["boxes/difficult"].numpy().flatten()

        # Filter out difficult boxes if keep_difficult is False
        if not self.keep_difficult:
            keep_mask = difficult == 0  # Keep non-difficult boxes
            boxes = boxes[keep_mask]
            labels = labels[keep_mask]

        iscrowd = torch.zeros(len(boxes), dtype=torch.int64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(batch):
    return tuple(zip(*batch))
