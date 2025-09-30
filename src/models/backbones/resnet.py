# -*- coding: utf-8 -*-
"""ResNet backbone utilities shared across detection models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

__all__ = [
    "ResNet",
    "ResNetBackboneConfig",
    "BackboneWithFPN",
    "build_resnet50_fpn_backbone",
    "RESNET_FPN_FEATMAP_NAMES",
]


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1×1 convolution from the reference notebook implementation."""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3×3 convolution with padding, matching the notebook helper."""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    """ResNet bottleneck block mirroring the lab reference."""

    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ===== STUDENT TODO: Implement Bottleneck forward pass =====
        # Hint: Follow the ResNet bottleneck structure:
        # 1. Apply first 1x1 conv + batch norm + ReLU
        # 2. Apply 3x3 conv + batch norm + ReLU
        # 3. Apply second 1x1 conv + batch norm (no ReLU yet)
        # 4. Add skip connection (identity or downsample if needed)
        # 5. Apply final ReLU activation
        # Remember to handle the downsample path when stride > 1
        raise NotImplementedError("Bottleneck.forward() not implemented")
        # =============================================================


def _make_block(inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
    layers: List[nn.Module] = [Bottleneck(inplanes, planes, stride)]
    outplanes = planes * Bottleneck.expansion
    for _ in range(1, blocks):
        layers.append(Bottleneck(outplanes, planes))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """Minimal ResNet-50 backbone identical to the lab notebook version."""

    def __init__(self, layers: Iterable[int] = (3, 4, 6, 3), num_classes: int = 1000) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        expansion = Bottleneck.expansion
        self.layer1 = _make_block(self.inplanes, 64, layers[0])
        self.layer2 = _make_block(64 * expansion, 128, layers[1], stride=2)
        self.layer3 = _make_block(128 * expansion, 256, layers[2], stride=2)
        self.layer4 = _make_block(256 * expansion, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ===== STUDENT TODO: Implement ResNet forward pass =====
        # Hint: Follow the standard ResNet-50 architecture:
        # 1. Apply initial conv1 + batch norm + ReLU + maxpool
        # 2. Pass through layer1, layer2, layer3, layer4 in sequence
        # 3. Apply global average pooling and flatten
        # 4. Apply final fully connected layer
        # This should match the torchvision ResNet-50 structure
        raise NotImplementedError("ResNet.forward() not implemented")
        # ========================================================


class BackboneWithFPN(nn.Module):
    """Wrap the ResNet backbone with an FPN, matching the notebook flow."""

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
            norm_layer=nn.BatchNorm2d,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        # ===== STUDENT TODO: Implement BackboneWithFPN forward pass =====
        # Hint: Combine backbone features with FPN:
        # 1. Extract intermediate features using self.body (IntermediateLayerGetter)
        # 2. Pass features through FPN (self.fpn) to create feature pyramid
        # 3. Return the FPN output (OrderedDict of multi-scale features)
        # This creates the feature pyramid needed for multi-scale detection
        raise NotImplementedError("BackboneWithFPN.forward() not implemented")
        # =================================================================


@dataclass
class ResNetBackboneConfig:
    """Configuration for constructing the ResNet+FPN backbone."""

    pretrained: bool = False
    trainable_layers: int = 6
    out_channels: int = 256
    weights: Optional[ResNet50_Weights] = None
    weights_path: Optional[str] = None


def _load_pretrained_weights(backbone: ResNet, config: ResNetBackboneConfig) -> None:
    if not config.pretrained:
        return

    state_dict = None
    if config.weights_path:
        state_dict = torch.load(config.weights_path, map_location="cpu")
    elif config.weights:
        try:
            state_dict = resnet50(weights=config.weights).state_dict()
        except Exception:
            state_dict = resnet50(weights=None).state_dict()

    if state_dict is not None:
        backbone.load_state_dict(state_dict)


def _freeze_backbone_layers(backbone: ResNet, trainable_layers: int) -> None:
    layers = ["layer4", "layer3", "layer2", "layer1", "conv1", "bn1"]
    max_layers = len(layers)
    if trainable_layers < 0 or trainable_layers > max_layers:
        raise ValueError(f"trainable_layers must be in [0, {max_layers}]")

    layers_to_train = layers[:trainable_layers] if trainable_layers > 0 else []

    for name, parameter in backbone.named_parameters():
        parameter.requires_grad = any(name.startswith(layer) for layer in layers_to_train)


def build_resnet50_fpn_backbone(config: Optional[ResNetBackboneConfig] = None) -> BackboneWithFPN:
    """Instantiate a ResNet-50 FPN backbone."""
    # ===== STUDENT TODO: Implement ResNet-FPN backbone construction =====
    # Hint: Build complete ResNet+FPN backbone:
    # 1. Create ResNet instance and handle pretrained weights/freezing
    # 2. Define return_layers dict to extract features from layer1-4
    # 3. Calculate in_channels_list for each FPN level
    #    (hint: use backbone.fc.in_features to determine channel progression)
    # 4. Create and return BackboneWithFPN with all components
    # This integrates ResNet feature extraction with FPN multi-scale features
    print(config)
    raise NotImplementedError("build_resnet50_fpn_backbone() not implemented")
    # ===================================================================


RESNET_FPN_FEATMAP_NAMES = ("0", "1", "2", "3")