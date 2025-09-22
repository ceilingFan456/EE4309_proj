# EE4309 Object Detection - Test Suite

This test suite evaluates student implementations of core object detection components.

## Test Structure

### Core Component Tests

1. **`test_resnet_backbone.py`** - Tests ResNet backbone implementation
   - `TestBottleneckBlock`: Tests ResNet bottleneck block forward pass
   - `TestResNet`: Tests complete ResNet model forward pass
   - `TestResNetFPNBackbone`: Tests ResNet+FPN backbone construction

2. **`test_vit_backbone.py`** - Tests Vision Transformer backbone implementation
   - `TestAttention`: Tests multi-head attention mechanism
   - `TestTransformerBlock`: Tests Transformer block forward pass
   - `TestViT`: Tests complete ViT model forward pass
   - `TestViTBackboneBuilders`: Tests ViT backbone builder functions

3. **`test_faster_rcnn.py`** - Tests Faster R-CNN detector assembly
   - `TestFasterRCNNAssembly`: Tests detector assembly function
   - `TestResNet50FasterRCNN`: Tests complete ResNet-50 detector
   - `TestDetectorConfig`: Tests detector configuration

4. **`test_training_engine.py`** - Tests training loop implementation
   - `TestTrainingStepIntegration`: Tests training step logic
   - `TestEvaluationIntegration`: Tests mAP evaluation logic
   - `TestTrainingComponents`: Tests individual training components

5. **`test_model_integration.py`** - Tests end-to-end model integration
   - `TestModelIntegration`: Tests complete model construction
   - `TestDatasetIntegration`: Tests dataset loading
   - `TestEndToEndPipeline`: Tests full pipeline integration
   - `TestImplementationStatus`: Reports implementation status

## Running Tests

### Run All Tests
```bash
make test
```

### Run Specific Test Files
```bash
python -m pytest tests/test_resnet_backbone.py -v
python -m pytest tests/test_vit_backbone.py -v
python -m pytest tests/test_faster_rcnn.py -v
python -m pytest tests/test_training_engine.py -v
python -m pytest tests/test_model_integration.py -v
```

### Run with Implementation Status Report
```bash
python -m pytest tests/test_model_integration.py::TestImplementationStatus::test_implementation_status_report -v -s
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Focus on specific implementation requirements

### Integration Tests
- Test component interactions
- Verify end-to-end functionality
- Check data flow between components

### Structure Tests
- Verify code can be imported
- Check class/function signatures
- Validate expected attributes exist

## Expected Test Behavior

### Before Implementation
- Tests will fail with `NotImplementedError`
- Some tests may be skipped due to missing dependencies
- Implementation status report shows "❌ NOT IMPLEMENTED"

### After Implementation
- Tests should pass with correct implementations
- All components should integrate properly
- Implementation status report shows "✅ IMPLEMENTED"

## Grading Criteria

The test suite evaluates:

1. **Correctness** (60%)
   - Proper implementation of forward passes
   - Correct tensor shapes and data flow
   - Accurate model assembly

2. **Integration** (25%)
   - Components work together properly
   - Model can be constructed and used
   - Training loop executes correctly

3. **Code Quality** (15%)
   - Proper error handling
   - Expected class/method structure
   - Meaningful variable usage

## Common Issues

### Import Errors
- Check that all files are in correct locations
- Ensure `src/` is in Python path
- Verify all required dependencies are installed

### Shape Mismatches
- Check tensor dimensions in forward passes
- Verify proper reshaping and permutations
- Ensure batch dimensions are handled correctly

### NotImplementedError
- Indicates student code needs to be completed
- Replace `raise NotImplementedError(...)` with actual implementation
- Follow the hints provided in comments

### CUDA Issues
- Tests are configured to run on CPU
- Mixed precision training uses CPU-compatible settings
- GPU-specific code should have CPU fallbacks

## Testing Best Practices

- Run tests frequently during development
- Focus on one component at a time
- Use implementation status report to track progress
- Read test failure messages carefully for debugging hints