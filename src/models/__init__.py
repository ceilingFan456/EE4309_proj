from __future__ import annotations

from typing import Callable, Dict

from .faster_rcnn import ResNetBackboneConfig, get_resnet50_fasterrcnn_model
from .vit_frcnn import ViTBackboneConfig, get_vit_fasterrcnn_model


_MODEL_REGISTRY: Dict[str, Callable[..., object]] = {
    "vit": get_vit_fasterrcnn_model,
    "vitdet": get_vit_fasterrcnn_model,
    "cnn": get_resnet50_fasterrcnn_model,
    "resnet50": get_resnet50_fasterrcnn_model,
    "fasterrcnn_resnet50": get_resnet50_fasterrcnn_model,
}


def build_model(name: str, /, **kwargs):
    """Factory helper to construct detection models by name."""

    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[key](**kwargs)


AVAILABLE_MODELS = tuple(sorted(_MODEL_REGISTRY))


__all__ = [
    "build_model",
    "get_resnet50_fasterrcnn_model",
    "get_vit_fasterrcnn_model",
    "ResNetBackboneConfig",
    "ViTBackboneConfig",
    "AVAILABLE_MODELS",
]
