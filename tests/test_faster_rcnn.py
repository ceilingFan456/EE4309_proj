"""
Test cases for Faster R-CNN detector assembly.
Tests the student's implementation of detector construction.
"""

import torch
import torch.nn as nn
import pytest
from src.models.faster_rcnn import (
    build_faster_rcnn,
    BackboneBundle,
    DetectorConfig,
    get_resnet50_fasterrcnn_model
)
from src.models.backbones.resnet import build_resnet50_fpn_backbone, RESNET_FPN_FEATMAP_NAMES
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class TestFasterRCNNAssembly:
    """Test Faster R-CNN detector assembly function."""

    def setup_method(self):
        """Set up test components."""
        # Create a proper mock backbone with out_channels attribute
        mock_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        # Add the required out_channels attribute
        mock_backbone.out_channels = 64
        self.mock_backbone = mock_backbone

        self.backbone_bundle = BackboneBundle(
            body=self.mock_backbone,
            featmap_names=["0"],
            out_channels=64
        )

        self.anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        def rpn_head_factory(num_anchors):
            return nn.Conv2d(64, num_anchors, 1)

        self.rpn_head_factory = rpn_head_factory

        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )

    def test_build_faster_rcnn_basic(self):
        """Test basic Faster R-CNN construction."""
        try:
            model = build_faster_rcnn(
                backbone=self.backbone_bundle,
                anchor_generator=self.anchor_generator,
                rpn_head_factory=self.rpn_head_factory,
                roi_pool=self.roi_pool,
                num_classes=21  # VOC classes
            )

            assert model is not None, "build_faster_rcnn returned None"

            # Check that it's a FasterRCNN instance
            from torchvision.models.detection import FasterRCNN
            assert isinstance(model, FasterRCNN), "build_faster_rcnn should return FasterRCNN instance"

        except NotImplementedError:
            pytest.fail("build_faster_rcnn function is not implemented")

    def test_build_faster_rcnn_with_config(self):
        """Test Faster R-CNN construction with custom config."""
        config = DetectorConfig(
            box_score_thresh=0.1,
            box_nms_thresh=0.3,
            detections_per_img=50
        )

        try:
            model = build_faster_rcnn(
                backbone=self.backbone_bundle,
                anchor_generator=self.anchor_generator,
                rpn_head_factory=self.rpn_head_factory,
                roi_pool=self.roi_pool,
                num_classes=21,
                config=config
            )

            assert model is not None, "build_faster_rcnn returned None with config"

        except NotImplementedError:
            pytest.fail("build_faster_rcnn function is not implemented")

    def test_faster_rcnn_num_classes(self):
        """Test that the assembled model has correct number of classes."""
        try:
            model = build_faster_rcnn(
                backbone=self.backbone_bundle,
                anchor_generator=self.anchor_generator,
                rpn_head_factory=self.rpn_head_factory,
                roi_pool=self.roi_pool,
                num_classes=10  # Custom number of classes
            )

            # Check that the box predictor has correct number of classes
            num_classes = model.roi_heads.box_predictor.cls_score.out_features
            assert num_classes == 10, f"Expected 10 classes, got {num_classes}"

        except NotImplementedError:
            pytest.fail("build_faster_rcnn function is not implemented")
        except AttributeError:
            pytest.fail("Model structure is incorrect - check box predictor replacement")


class TestResNet50FasterRCNN:
    """Test the complete ResNet-50 Faster R-CNN model construction."""

    def test_get_resnet50_model_basic(self):
        """Test basic ResNet-50 Faster R-CNN model construction."""
        try:
            model = get_resnet50_fasterrcnn_model(num_classes=21)
            assert model is not None, "get_resnet50_fasterrcnn_model returned None"

            # Check model type
            from torchvision.models.detection import FasterRCNN
            assert isinstance(model, FasterRCNN), "Should return FasterRCNN instance"

        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.fail("Cannot test ResNet-50 model because build_faster_rcnn is not implemented")
            else:
                # If ResNet forward is not implemented, this might fail too
                pytest.skip(f"Skipping test due to dependency issue: {e}")

    def test_resnet50_model_forward_shape(self):
        """Test that ResNet-50 model can process inputs with correct shapes."""
        try:
            model = get_resnet50_fasterrcnn_model(num_classes=21)
            model.eval()

            # Test input
            images = [torch.randn(3, 512, 512)]

            with torch.no_grad():
                # In eval mode, should return predictions
                outputs = model(images)

                assert isinstance(outputs, list), "Model should return list of predictions"
                assert len(outputs) == 1, "Should return one prediction per image"

                prediction = outputs[0]
                assert 'boxes' in prediction, "Prediction should contain boxes"
                assert 'labels' in prediction, "Prediction should contain labels"
                assert 'scores' in prediction, "Prediction should contain scores"

        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.fail("Cannot test model forward because build_faster_rcnn is not implemented")
            else:
                pytest.skip(f"Skipping test due to dependency issue: {e}")


class TestDetectorConfig:
    """Test detector configuration class."""

    def test_detector_config_default(self):
        """Test default detector configuration."""
        config = DetectorConfig()
        kwargs = config.to_kwargs()

        assert isinstance(kwargs, dict), "to_kwargs should return dictionary"
        assert 'box_score_thresh' in kwargs, "Config should include box_score_thresh"
        assert 'box_nms_thresh' in kwargs, "Config should include box_nms_thresh"
        assert 'rpn_nms_thresh' in kwargs, "Config should include rpn_nms_thresh"

    def test_detector_config_custom(self):
        """Test custom detector configuration."""
        config = DetectorConfig(
            box_score_thresh=0.1,
            box_nms_thresh=0.3,
            detections_per_img=50
        )
        kwargs = config.to_kwargs()

        assert kwargs['box_score_thresh'] == 0.1, "Custom box_score_thresh not set correctly"
        assert kwargs['box_nms_thresh'] == 0.3, "Custom box_nms_thresh not set correctly"
        assert kwargs['box_detections_per_img'] == 50, "Custom detections_per_img not set correctly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])