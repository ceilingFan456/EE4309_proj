#!/usr/bin/env bash
set -euo pipefail

# Download sample images for inference testing
SAMPLE_DIR="data/sample_images"
mkdir -p "$SAMPLE_DIR"

echo "📥 Downloading sample images to $SAMPLE_DIR"

# Download sample images from public datasets

echo "1. Downloading image with person..."
curl -L "https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg" \
     -o "$SAMPLE_DIR/person.jpg" --fail --silent || echo "   ⚠️  Download failed, please add manually"

echo "2. Downloading image with dog..."
curl -L "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg" \
     -o "$SAMPLE_DIR/dog.jpg" --fail --silent || echo "   ⚠️  Download failed, please add manually"

echo "3. Downloading image with horses..."
curl -L "https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg" \
     -o "$SAMPLE_DIR/horses.jpg" --fail --silent || echo "   ⚠️  Download failed, please add manually"

echo "4. Downloading image with eagle..."
curl -L "https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg" \
     -o "$SAMPLE_DIR/eagle.jpg" --fail --silent || echo "   ⚠️  Download failed, please add manually"

# Check download results
echo ""
echo "📊 Download results:"
if [ -d "$SAMPLE_DIR" ]; then
    file_count=$(find "$SAMPLE_DIR" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
    if [ "$file_count" -gt 0 ]; then
        echo "✅ Successfully downloaded $file_count sample images"
        echo "📁 Image list:"
        ls -lh "$SAMPLE_DIR"/*.{jpg,jpeg,png} 2>/dev/null || true
        echo ""
        echo "🚀 Now you can run inference test:"
        echo "   bash scripts/infer.sh runs/frcnn_r50fpn_voc07/best.pt data/sample_images/"
    else
        echo "⚠️  No images downloaded successfully. Please manually add images containing:"
        echo "   person, car, dog, cat, bicycle, bird, etc."
    fi
else
    echo "❌ Failed to create sample images directory"
fi

echo ""
echo "💡 Tips:"
echo "1. You can also copy your own images to $SAMPLE_DIR directory"
echo "2. Supported formats: jpg, jpeg, png"
echo "3. Images should preferably contain objects from VOC dataset's 20 classes"