import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Get image width before flipping
            if hasattr(image, 'width'):
                # PIL Image
                width = image.width
            elif isinstance(image, torch.Tensor):
                # Tensor image (C, H, W)
                width = image.shape[-1]
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            image = F.hflip(image)
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if "boxes" in target:
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        if "labels" in target:
            target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        return image, target
