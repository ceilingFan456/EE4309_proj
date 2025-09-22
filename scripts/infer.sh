#!/usr/bin/env bash
set -euo pipefail
WEIGHTS=${1:-}
INPUT=${2:-data/sample_images}
MODEL_INPUT=${3:-}

MODEL=""
if [ -n "$MODEL_INPUT" ]; then
  MODEL=$(echo "$MODEL_INPUT" | tr '[:upper:]' '[:lower:]')
fi

MODEL_TAG=$MODEL
if [ -z "$MODEL_TAG" ]; then
  MODEL_TAG="vit"
fi

if [ -z "$WEIGHTS" ]; then
  WEIGHTS="runs/${MODEL_TAG}_voc07/best.pt"
fi

# Check if input path exists
if [ ! -e "$INPUT" ]; then
    echo "❌ Error: Input path '$INPUT' does not exist!"
    echo ""
    echo "Please ensure:"
    echo "1. Sample images exist in data/sample_images/ (default)"
    echo "2. Or VOC dataset is downloaded to data/VOCdevkit/VOC2007/JPEGImages/"
    echo "3. Or provide path to your own images"
    echo ""
    echo "Usage: $0 [weights_path] [input_path]"
    echo "Example: $0 runs/vit_voc07/best.pt data/sample_images/"
    exit 1
fi

# Check if weights file exists
if [ ! -f "$WEIGHTS" ]; then
    echo "❌ Error: Weights file '$WEIGHTS' does not exist!"
    echo "Please train the model first: make train"
    exit 1
fi

echo "🚀 Running inference..."
echo "Weights: $WEIGHTS"
echo "Input: $INPUT"
echo "Model (requested/default): $MODEL_TAG"

CMD=(python3 -m src.engine.inference \
  --weights "$WEIGHTS" \
  --input "$INPUT" \
  --score-thr 0.5 \
  --output runs/infer_vis)

if [ -n "$MODEL" ]; then
  CMD+=(--model "$MODEL")
fi

"${CMD[@]}"

echo "✅ Inference completed! Check results in runs/infer_vis/"
