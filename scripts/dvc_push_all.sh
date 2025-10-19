#!/bin/bash
# Push DVC data to all remotes
# Usage: ./scripts/dvc_push_all.sh

echo "📤 Pushing DVC data to all remotes..."
echo ""

FAILED=0

# Push to Backblaze
echo "🔵 Pushing to Backblaze..."
if dvc push -r backblaze; then
    echo "✅ Backblaze push successful"
else
    echo "❌ Backblaze push failed"
    FAILED=1
fi

echo ""

# Push to DagsHub
echo "🟣 Pushing to DagsHub..."
if dvc push -r dagshub; then
    echo "✅ DagsHub push successful"
else
    echo "❌ DagsHub push failed"
    FAILED=1
fi

echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All pushes successful!"
    exit 0
else
    echo "⚠️  Some pushes failed. Check the output above."
    exit 1
fi
