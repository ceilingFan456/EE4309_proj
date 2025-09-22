"""
Test cases for ResNet backbone implementation.
Tests the student's implementation of core ResNet components.
"""

import torch
import torch.nn as nn
import pytest
from src.models.backbones.resnet import Bottleneck, ResNet, build_resnet50_fpn_backbone


class TestBottleneckBlock:
    """Test ResNet Bottleneck block implementation."""

    def test_bottleneck_forward_basic(self):
        """Test basic bottleneck forward pass."""
        # Create a bottleneck block
        bottleneck = Bottleneck(inplanes=64, planes=64, stride=1)

        # Test input
        x = torch.randn(2, 64, 56, 56)

        try:
            output = bottleneck(x)
            # Check output shape - channels should be planes * expansion (64 * 4 = 256)
            expected_shape = (2, 256, 56, 56)  # planes * expansion
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

            # Check that output contains meaningful values (not just zeros/ones)
            assert torch.std(output) > 0.1, "Output appears to be constant - check implementation"

        except NotImplementedError:
            pytest.fail("Bottleneck forward method is not implemented")

    def test_bottleneck_forward_with_stride(self):
        """Test bottleneck forward pass with stride > 1."""
        # Create bottleneck with stride=2 (should trigger downsampling)
        bottleneck = Bottleneck(inplanes=64, planes=128, stride=2)

        # Test input
        x = torch.randn(2, 64, 56, 56)

        try:
            output = bottleneck(x)
            # Check output shape - spatial dims should be halved, channels increased
            expected_shape = (2, 128 * 4, 28, 28)  # planes * expansion
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

        except NotImplementedError:
            pytest.fail("Bottleneck forward method is not implemented")

    def test_bottleneck_has_downsample(self):
        """Test that downsample is created when needed."""
        # This should create downsample (different planes)
        bottleneck = Bottleneck(inplanes=64, planes=128, stride=2)
        assert hasattr(bottleneck, 'downsample'), "Downsample should be created when stride > 1"

        # This should also create downsample (inplanes != planes * expansion)
        bottleneck = Bottleneck(inplanes=64, planes=64, stride=1)  # 64 != 64 * 4
        assert hasattr(bottleneck, 'downsample'), "Downsample should be created when channel mismatch"

        # This should not create downsample
        bottleneck = Bottleneck(inplanes=256, planes=64, stride=1)  # 256 == 64 * 4
        assert hasattr(bottleneck, 'downsample') == False, "Downsample should not be created when channels match"


class TestResNet:
    """Test ResNet model implementation."""

    def test_resnet_forward_basic(self):
        """Test basic ResNet forward pass."""
        model = ResNet()
        x = torch.randn(2, 3, 224, 224)

        try:
            output = model(x)
            # Check output shape for classification
            assert output.shape == (2, 1000), f"Expected shape (2, 1000), got {output.shape}"

            # Check that output contains meaningful values
            assert torch.std(output) > 0.1, "Output appears to be constant - check implementation"

        except NotImplementedError:
            pytest.fail("ResNet forward method is not implemented")

    def test_resnet_forward_different_input_size(self):
        """Test ResNet with different input sizes."""
        model = ResNet()

        # Test with different input size
        x = torch.randn(1, 3, 512, 512)

        try:
            output = model(x)
            # Output should still be classification size regardless of input size
            assert output.shape == (1, 1000), f"Expected shape (1, 1000), got {output.shape}"

        except NotImplementedError:
            pytest.fail("ResNet forward method is not implemented")

    def test_resnet_layers_exist(self):
        """Test that all required layers exist in ResNet."""
        model = ResNet()

        required_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']

        for layer_name in required_layers:
            assert hasattr(model, layer_name), f"ResNet missing required layer: {layer_name}"


class TestResNetFPNBackbone:
    """Test ResNet FPN backbone construction."""

    def test_fpn_backbone_construction(self):
        """Test that FPN backbone can be constructed."""
        try:
            backbone = build_resnet50_fpn_backbone()
            assert backbone is not None, "Failed to build ResNet FPN backbone"

            # Test forward pass through backbone
            x = torch.randn(1, 3, 512, 512)
            features = backbone(x)

            # Should return multiple feature levels
            assert isinstance(features, dict), "Backbone should return dict of features"
            assert len(features) > 1, "Backbone should return multiple feature levels"

        except Exception as e:
            # If ResNet forward is not implemented, the backbone construction might fail
            if "not implemented" in str(e).lower():
                pytest.fail("Cannot test FPN backbone because ResNet forward is not implemented")
            else:
                raise e


if __name__ == "__main__":
    pytest.main([__file__, "-v"])