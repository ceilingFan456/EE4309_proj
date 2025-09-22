#!/usr/bin/env bash
set -euo pipefail

# Default parameters
WEIGHTS=${1:-}
IMAGE_SET=${2:-test}
SUBSET_SIZE=${3:-500}
MODEL_INPUT=${4:-}

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

# Check if weights file exists
if [ ! -f "$WEIGHTS" ]; then
    echo "❌ Error: Weights file '$WEIGHTS' does not exist!"
    echo ""
    echo "Please ensure:"
    echo "1. You have trained the model: make train"
    echo "2. Or provide path to existing weights"
    echo ""
    echo "Usage: $0 [weights_path] [image_set] [subset_size] [model]"
    echo "Example: $0 runs/${MODEL_TAG}_voc07/best.pt test 500 ${MODEL_TAG}"
    exit 1
fi

echo "🚀 Starting evaluation..."
echo "Weights: $WEIGHTS"
echo "Image set: $IMAGE_SET"
echo "Subset size: $SUBSET_SIZE"
echo "Model (requested/default): $MODEL_TAG"
echo ""

CMD=(python3 -m src.engine.eval \
  --image-set "$IMAGE_SET" \
  --weights "$WEIGHTS" \
  --batch-size 1 \
  --num-workers 8 \
  --subset-size "$SUBSET_SIZE")

if [ -n "$MODEL" ]; then
  CMD+=(--model "$MODEL")
fi

"${CMD[@]}"

echo "✅ Evaluation completed!"
