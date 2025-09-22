"""
Test cases for Vision Transformer backbone implementation.
Tests the student's implementation of core ViT components.
"""

import torch
import torch.nn as nn
import pytest
from src.models.backbones.vit import Attention, Block, ViT, build_vit_backbone, build_vit_fpn_backbone


class TestAttention:
    """Test Vision Transformer Attention mechanism."""

    def test_attention_forward_basic(self):
        """Test basic attention forward pass."""
        # Create attention module
        attention = Attention(dim=384, num_heads=6, input_size=(32, 32))

        # Test input (B, H, W, C format for ViT)
        x = torch.randn(2, 32, 32, 384)

        try:
            output = attention(x)
            # Check output shape should match input
            assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

            # Check that output contains meaningful values
            assert torch.std(output) > 0.01, "Output appears to be constant - check implementation"

        except NotImplementedError:
            pytest.fail("Attention forward method is not implemented")

    def test_attention_different_heads(self):
        """Test attention with different number of heads."""
        attention = Attention(dim=384, num_heads=12, input_size=(16, 16))
        x = torch.randn(1, 16, 16, 384)

        try:
            output = attention(x)
            assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

        except NotImplementedError:
            pytest.fail("Attention forward method is not implemented")

    def test_attention_parameters_exist(self):
        """Test that attention has required parameters."""
        attention = Attention(dim=384, num_heads=6, input_size=(32, 32))

        required_params = ['qkv', 'proj', 'rel_pos_h', 'rel_pos_w']
        for param_name in required_params:
            assert hasattr(attention, param_name), f"Attention missing required parameter: {param_name}"


class TestTransformerBlock:
    """Test Vision Transformer Block implementation."""

    def test_block_forward_basic(self):
        """Test basic transformer block forward pass."""
        # Create transformer block
        block = Block(dim=384, num_heads=6, input_size=(32, 32))

        # Test input
        x = torch.randn(2, 32, 32, 384)

        try:
            output = block(x)
            # Check output shape should match input
            assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

            # Check that output contains meaningful values
            assert torch.std(output) > 0.01, "Output appears to be constant - check implementation"

        except NotImplementedError:
            pytest.fail("Block forward method is not implemented")

    def test_block_with_window_attention(self):
        """Test transformer block with windowed attention."""
        block = Block(dim=384, num_heads=6, window_size=8, input_size=(32, 32))
        x = torch.randn(1, 32, 32, 384)

        try:
            output = block(x)
            assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

        except NotImplementedError:
            pytest.fail("Block forward method is not implemented")

    def test_block_residual_connection(self):
        """Test that block implements residual connections correctly."""
        block = Block(dim=384, num_heads=6, input_size=(16, 16))

        # Create input that should have some residual effect
        x = torch.randn(1, 16, 16, 384)

        try:
            output = block(x)
            # Output should be different from input due to transformations
            assert not torch.allclose(output, x, atol=1e-3), "Output is too similar to input - check residual connections"

        except NotImplementedError:
            pytest.fail("Block forward method is not implemented")


class TestViT:
    """Test Vision Transformer model implementation."""

    def test_vit_forward_basic(self):
        """Test basic ViT forward pass."""
        vit = ViT(img_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6)
        x = torch.randn(2, 3, 224, 224)

        try:
            output = vit(x)
            # Should return a dictionary with feature name as key
            assert isinstance(output, dict), "ViT should return a dictionary"
            assert len(output) == 1, "ViT should return one feature by default"

            # Check output feature shape
            feature_key = list(output.keys())[0]
            feature_tensor = output[feature_key]

            # Should be in BCHW format after permutation
            batch_size, channels, height, width = feature_tensor.shape
            assert batch_size == 2, f"Expected batch size 2, got {batch_size}"
            assert channels == 384, f"Expected channels 384, got {channels}"

        except NotImplementedError:
            pytest.fail("ViT forward method is not implemented")

    def test_vit_different_image_size(self):
        """Test ViT with different image sizes."""
        vit = ViT(img_size=512, patch_size=16, embed_dim=384, depth=6, num_heads=6)
        x = torch.randn(1, 3, 512, 512)

        try:
            output = vit(x)
            assert isinstance(output, dict), "ViT should return a dictionary"

            feature_key = list(output.keys())[0]
            feature_tensor = output[feature_key]

            # Check that spatial dimensions match patch grid
            expected_spatial = 512 // 16  # 32
            assert feature_tensor.shape[2] == expected_spatial, f"Expected height {expected_spatial}, got {feature_tensor.shape[2]}"
            assert feature_tensor.shape[3] == expected_spatial, f"Expected width {expected_spatial}, got {feature_tensor.shape[3]}"

        except NotImplementedError:
            pytest.fail("ViT forward method is not implemented")

    def test_vit_components_exist(self):
        """Test that ViT has all required components."""
        vit = ViT(embed_dim=384, depth=6, num_heads=6)

        required_components = ['patch_embed', 'pos_embed', 'blocks']
        for component_name in required_components:
            assert hasattr(vit, component_name), f"ViT missing required component: {component_name}"

        # Check that we have the right number of blocks
        assert len(vit.blocks) == 6, f"Expected 6 transformer blocks, got {len(vit.blocks)}"


class TestViTBackboneBuilders:
    """Test ViT backbone builder functions."""

    def test_build_vit_backbone(self):
        """Test building basic ViT backbone."""
        try:
            vit = build_vit_backbone()
            assert vit is not None, "Failed to build ViT backbone"

            # Test a forward pass
            x = torch.randn(1, 3, 512, 512)
            output = vit(x)
            assert isinstance(output, dict), "ViT backbone should return dict"

        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.fail("Cannot test ViT backbone builder because ViT forward is not implemented")
            else:
                raise e

    def test_build_vit_fpn_backbone(self):
        """Test building ViT with FPN backbone."""
        try:
            backbone = build_vit_fpn_backbone()
            assert backbone is not None, "Failed to build ViT FPN backbone"

            # Test a forward pass
            x = torch.randn(1, 3, 512, 512)
            features = backbone(x)

            assert isinstance(features, dict), "ViT FPN backbone should return dict of features"
            assert len(features) > 1, "ViT FPN backbone should return multiple feature levels"

        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.fail("Cannot test ViT FPN backbone because ViT forward is not implemented")
            else:
                raise e


if __name__ == "__main__":
    pytest.main([__file__, "-v"])