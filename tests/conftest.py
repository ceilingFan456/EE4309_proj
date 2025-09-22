"""
Pytest configuration and fixtures for EE4309 Object Detection tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Force CPU for tests to avoid CUDA issues
    torch.set_default_device("cpu")

    yield

    # Cleanup after all tests
    pass


@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_batch_size():
    """Provide small batch size for testing."""
    return 2


@pytest.fixture
def test_image_tensor():
    """Provide test image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def test_feature_tensor():
    """Provide test feature tensor for ViT."""
    return torch.randn(1, 32, 32, 384)


@pytest.fixture
def mock_detection_targets():
    """Provide mock detection targets."""
    return [{
        'boxes': torch.tensor([[10, 10, 50, 50], [20, 20, 80, 80]], dtype=torch.float32),
        'labels': torch.tensor([1, 2], dtype=torch.int64)
    }]


@pytest.fixture
def mock_detection_predictions():
    """Provide mock detection predictions."""
    return [{
        'boxes': torch.tensor([[12, 12, 48, 48], [18, 18, 82, 82]], dtype=torch.float32),
        'labels': torch.tensor([1, 2], dtype=torch.int64),
        'scores': torch.tensor([0.9, 0.8], dtype=torch.float32)
    }]


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle skips appropriately."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark potentially slow tests
        if any(word in item.nodeid.lower() for word in ["training", "model", "forward"]):
            item.add_marker(pytest.mark.slow)


# Custom assertion helpers
class TestHelpers:
    """Helper methods for tests."""

    @staticmethod
    def assert_tensor_shape(tensor, expected_shape, name="tensor"):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} != expected {expected_shape}"

    @staticmethod
    def assert_tensor_not_constant(tensor, threshold=0.01, name="tensor"):
        """Assert tensor values are not constant."""
        assert torch.std(tensor) > threshold, f"{name} appears to be constant (std={torch.std(tensor):.6f})"

    @staticmethod
    def assert_gradients_flow(tensor, name="tensor"):
        """Assert tensor can compute gradients."""
        if tensor.requires_grad:
            assert tensor.grad_fn is not None or tensor.is_leaf, f"{name} should be able to compute gradients"


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers