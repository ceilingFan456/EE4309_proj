# -*- coding: utf-8 -*-
"""Backbone modules for detection models."""

from .vit import (
    ShapeSpec,
    ViTBackboneConfig,
    PatchEmbed,
    Attention,
    Block,
    ViT,
    SimpleFeaturePyramid,
    LastLevelMaxPool,
    build_vit_backbone,
    build_vit_fpn_backbone,
)
from .resnet import (
    ResNet,
    ResNetBackboneConfig,
    BackboneWithFPN,
    build_resnet50_fpn_backbone,
    RESNET_FPN_FEATMAP_NAMES,
)

__all__ = [
    "ShapeSpec",
    "ViTBackboneConfig",
    "PatchEmbed",
    "Attention",
    "Block",
    "ViT",
    "SimpleFeaturePyramid",
    "LastLevelMaxPool",
    "build_vit_backbone",
    "build_vit_fpn_backbone",
    "ResNet",
    "ResNetBackboneConfig",
    "BackboneWithFPN",
    "build_resnet50_fpn_backbone",
    "RESNET_FPN_FEATMAP_NAMES",
]
