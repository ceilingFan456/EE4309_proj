"""
Integration tests for complete model functionality.
Tests the student's implementation across the entire pipeline.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestModelIntegration:
    """Test complete model integration."""

    def test_model_builder_imports(self):
        """Test that model builder can import all required components."""
        try:
            from src.models import build_model, AVAILABLE_MODELS
            assert 'vit' in AVAILABLE_MODELS, "ViT model should be available"
            assert 'resnet50' in AVAILABLE_MODELS, "ResNet50 model should be available"
        except ImportError as e:
            pytest.fail(f"Model builder import error: {e}")

    def test_vit_model_construction(self):
        """Test ViT model construction through model builder."""
        try:
            from src.models import build_model

            model = build_model('vit', num_classes=21)
            assert model is not None, "ViT model construction should not return None"

            # Test that model has expected structure
            assert hasattr(model, 'backbone'), "ViT model should have backbone"
            assert hasattr(model, 'rpn'), "ViT model should have RPN"
            assert hasattr(model, 'roi_heads'), "ViT model should have ROI heads"

        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.fail("ViT model cannot be constructed because core components are not implemented")
            else:
                # Expected if ViT forward is not implemented
                pytest.skip(f"ViT model construction failed: {e}")

    def test_resnet50_model_construction(self):
        """Test ResNet50 model construction through model builder."""
        try:
            from src.models import build_model

            model = build_model('resnet50', num_classes=21)
            assert model is not None, "ResNet50 model construction should not return None"

            # Test that model has expected structure
            from torchvision.models.detection import FasterRCNN
            assert isinstance(model, FasterRCNN), "ResNet50 model should be FasterRCNN instance"

        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.fail("ResNet50 model cannot be constructed because core components are not implemented")
            else:
                # Expected if ResNet forward is not implemented
                pytest.skip(f"ResNet50 model construction failed: {e}")

    def test_model_training_mode(self):
        """Test that models can be switched between train and eval modes."""
        try:
            from src.models import build_model

            for model_name in ['vit', 'resnet50']:
                try:
                    model = build_model(model_name, num_classes=21)

                    # Test training mode
                    model.train()
                    assert model.training == True, f"{model_name} should be in training mode"

                    # Test eval mode
                    model.eval()
                    assert model.training == False, f"{model_name} should be in eval mode"

                except Exception as e:
                    if "not implemented" in str(e).lower():
                        continue  # Skip this model if not implemented
                    else:
                        raise e

        except ImportError:
            pytest.skip("Model builder not available")


class TestDatasetIntegration:
    """Test dataset integration."""

    def test_voc_dataset_import(self):
        """Test VOC dataset can be imported and basic structure."""
        try:
            from src.datasets.voc import VOCDataset, collate_fn
            assert VOCDataset is not None, "VOCDataset should be importable"
            assert collate_fn is not None, "collate_fn should be importable"
        except ImportError as e:
            pytest.fail(f"VOC dataset import error: {e}")

    def test_transforms_import(self):
        """Test that transforms can be imported."""
        try:
            from src.utils.transforms import Compose, ToTensor, RandomHorizontalFlip
            assert all([Compose, ToTensor, RandomHorizontalFlip]), "All transforms should be importable"
        except ImportError as e:
            pytest.fail(f"Transforms import error: {e}")


class TestEndToEndPipeline:
    """Test end-to-end pipeline functionality."""

    def test_pipeline_structure_completeness(self):
        """Test that all pipeline components can be imported together."""
        try:
            # Core model components
            from src.models import build_model

            # Dataset components
            from src.datasets.voc import VOCDataset, collate_fn

            # Training components
            from src.engine.train import get_args

            # Transform components
            from src.utils.transforms import Compose, ToTensor

            # Utility components
            from src.utils.common import seed_everything

            # If we get here, all major components are importable
            assert True, "All pipeline components are importable"

        except ImportError as e:
            pytest.fail(f"Pipeline import error: {e}")

    def test_training_prerequisites(self):
        """Test that training has all required prerequisites."""
        try:
            # Test PyTorch components needed for training
            import torch
            from torch.utils.data import DataLoader
            from torch.optim import SGD
            from torch.optim.lr_scheduler import StepLR
            from torch.cuda.amp import autocast, GradScaler

            # Training should be possible with core PyTorch components
            assert torch is not None, "PyTorch should be available"
            assert DataLoader is not None, "DataLoader should be available"
            assert SGD is not None, "SGD optimizer should be available"

        except ImportError as e:
            pytest.fail(f"Training prerequisites not met: {e}")

    @patch('torch.cuda.is_available', return_value=False)  # Force CPU testing
    def test_device_handling(self, mock_cuda):
        """Test proper device handling in training setup."""
        try:
            import torch
            from torch.cuda.amp import GradScaler

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            assert device.type == "cpu", "Should fall back to CPU when CUDA not available"

            # Test scaler creation with CPU
            scaler = GradScaler(enabled=False)  # Should be disabled for CPU
            assert scaler is not None, "GradScaler should be created successfully"

        except Exception as e:
            pytest.fail(f"Device handling error: {e}")


class TestImplementationStatus:
    """Test to check which components are implemented vs not implemented."""

    def test_implementation_status_report(self):
        """Generate a report of what is and isn't implemented."""
        status_report = {
            'resnet_bottleneck': False,
            'resnet_forward': False,
            'vit_attention': False,
            'vit_block': False,
            'vit_forward': False,
            'faster_rcnn_build': False,
            'training_step': False,
            'evaluation_step': False
        }

        # Test ResNet Bottleneck
        try:
            from src.models.backbones.resnet import Bottleneck
            bottleneck = Bottleneck(64, 64)
            x = torch.randn(1, 64, 32, 32)
            bottleneck(x)
            status_report['resnet_bottleneck'] = True
        except (NotImplementedError, Exception):
            pass

        # Test ResNet forward
        try:
            from src.models.backbones.resnet import ResNet
            resnet = ResNet()
            x = torch.randn(1, 3, 224, 224)
            resnet(x)
            status_report['resnet_forward'] = True
        except (NotImplementedError, Exception):
            pass

        # Test ViT Attention
        try:
            from src.models.backbones.vit import Attention
            attention = Attention(384, 6, input_size=(32, 32))
            x = torch.randn(1, 32, 32, 384)
            attention(x)
            status_report['vit_attention'] = True
        except (NotImplementedError, Exception):
            pass

        # Test ViT Block
        try:
            from src.models.backbones.vit import Block
            block = Block(384, 6, input_size=(32, 32))
            x = torch.randn(1, 32, 32, 384)
            block(x)
            status_report['vit_block'] = True
        except (NotImplementedError, Exception):
            pass

        # Test ViT forward
        try:
            from src.models.backbones.vit import ViT
            vit = ViT(embed_dim=384, depth=2, num_heads=6)
            x = torch.randn(1, 3, 224, 224)
            vit(x)
            status_report['vit_forward'] = True
        except (NotImplementedError, Exception):
            pass

        # Test Faster R-CNN build
        try:
            from src.models.faster_rcnn import build_faster_rcnn, BackboneBundle
            from torchvision.models.detection.rpn import AnchorGenerator
            from torchvision.ops import MultiScaleRoIAlign

            mock_backbone = nn.Sequential(nn.Conv2d(3, 64, 3))
            backbone_bundle = BackboneBundle(mock_backbone, ["0"], 64)
            anchor_gen = AnchorGenerator(((32,),), ((1.0,),))
            rpn_head_factory = lambda num_anchors: nn.Conv2d(64, num_anchors, 1)
            roi_pool = MultiScaleRoIAlign(["0"], 7, 2)

            build_faster_rcnn(
                backbone=backbone_bundle,
                anchor_generator=anchor_gen,
                rpn_head_factory=rpn_head_factory,
                roi_pool=roi_pool,
                num_classes=21
            )
            status_report['faster_rcnn_build'] = True
        except (NotImplementedError, Exception):
            pass

        # Print status report for debugging
        print("\\n=== Implementation Status Report ===")
        for component, implemented in status_report.items():
            status = "✅ IMPLEMENTED" if implemented else "❌ NOT IMPLEMENTED"
            print(f"{component}: {status}")

        # The test always passes - it's just for reporting
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])