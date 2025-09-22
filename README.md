# EE4309 Object Detection Project

This repository contains the lab project for **EE4309** at NUS. Students will implement and train object detection models using both traditional CNN backbones (ResNet) and modern Vision Transformer (ViT) architectures.

## Project Overview

### Learning Objectives
- Implement core components of Faster R-CNN object detection pipeline
- Understand and build Vision Transformer backbone for detection tasks
- Train and evaluate models on PASCAL VOC 2007 dataset
- Explore different training strategies and optimizations

### Key Components to Implement
Students are required to complete the following core modules:

1. **Model Architecture** (`src/models/`)
   - ResNet backbone implementation
   - Vision Transformer (ViT) backbone
   - Faster R-CNN detector assembly

2. **Training Engine** (`src/engine/train.py`)
   - Training loop with loss computation
   - Validation and mAP evaluation
   - Model checkpointing and logging

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended: 10GB+ VRAM)
- Conda or virtual environment

### Installation

1. **Setup environment**
   ```bash
   make setup
   conda activate ee4309-objdet
   ```

2. **Download sample data**
   ```bash
   make samples
   ```

### Training Models

**Train ResNet-50 Faster R-CNN:**
```bash
make train MODEL=resnet50 EPOCHS=12 BATCH_SIZE=2
```

**Train ViT-based detector:**
```bash
make train MODEL=vit EPOCHS=12 BATCH_SIZE=1
```

**Custom training:**

You can freely edit any parameter in `scripts/train.sh` as needed.
```bash
python src/engine/train.py --model vit --epochs 12 --batch-size 1 --lr 5e-3
```

### Evaluation

```bash
make eval WEIGHTS=runs/vit_voc07/best.pt MODEL=vit
```

### Inference

```bash
make infer INPUT=data/sample_images MODEL=vit WEIGHTS=runs/vit_voc07/best.pt
```

## Project Structure

```
EE4309_ViTDet/
├── src/
│   ├── models/              # Model architectures
│   │   ├── backbones/       # ResNet and ViT implementations
│   │   ├── faster_rcnn.py   # Detector assembly
│   │   └── vit_frcnn.py     # ViT detector wrapper
│   ├── engine/              # Training and evaluation
│   │   ├── train.py         # Main training script
│   │   ├── eval.py          # Evaluation script
│   │   └── inference.py     # Inference script
│   ├── datasets/            # Dataset loading
│   │   └── voc.py           # PASCAL VOC dataset
│   └── utils/               # Utilities
├── tests/                   # Unit tests
├── scripts/                 # Shell scripts
└── requirements.txt         # Dependencies
```

## Implementation Requirements

### Core Tasks (Required)

1. **Implement ResNet Backbone**
   - Complete `Bottleneck.forward()` method
   - Implement `ResNet.forward()` method
   - Add FPN integration

2. **Implement ViT Backbone**
   - Complete multi-head attention mechanism
   - Implement Transformer block forward pass
   - Add positional encoding and patch embedding

3. **Complete Training Engine**
   - Implement loss computation and backpropagation
   - Add mAP evaluation logic
   - Handle model saving and logging

4. **Detector Assembly**
   - Complete Faster R-CNN model building
   - Integrate backbone with detection head

### Advanced Tasks (Exploration)

Students are encouraged to explore:
- Different model architectures and sizes
- Training strategies (learning rate scheduling, data augmentation)
- Optimization techniques (mixed precision, gradient accumulation)
- Performance analysis and visualization

## Evaluation Metrics

Models will be evaluated using:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **Training efficiency**: Convergence speed and stability
- **Code quality**: Clean, well-documented implementation

## Development Tools

### Available Commands
```bash
make help          # Show all available commands
make test          # Run unit tests
make lint          # Check code style
make format        # Format code
make clean         # Clean outputs
```


## Submission Guidelines

### What to Submit
1. **Source code**: Complete implementation in the whole project (no need to upload the model weights)
2. **Training notebook**: Jupyter notebook with experiments and analysis

### Submission Format
- Submit as a zip file
- Include training logs and result visualizations in notebook
- Provide a brief markdown-formatted analysis of your approach


## Support

### Getting Help
- **TA Office Hours**: Check course website for schedule
- **Canvas Discussion**: Post questions for peer/TA support
- **Email TAs**: Guian & Junyuan for appointment booking

### Common Issues
- **CUDA out of memory**: Reduce batch size or image resolution
- **Slow training**: Enable mixed precision (`--no-amp` to disable)
- **Dataset loading**: Ensure internet connection for DeepLake

## Important Dates

- **Project Release**: 30 Sept, 12pm
- **Submission Deadline**: 24 Oct, 23:59
- **Lab Session**: 30 Sept, Tue, 12-4pm (E5-03-19)

## References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [ViTDet Paper](https://arxiv.org/abs/2203.16527)
- [PASCAL VOC Dataset](https://datasets.activeloop.ai/docs/ml/datasets/pascal-voc-2007-dataset/)

---

**Good luck with your implementation! 🚀**