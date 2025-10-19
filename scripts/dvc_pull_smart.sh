#!/bin/bash
# Smart DVC pull with automatic fallback
# Tries Backblaze first, falls back to DagsHub if it fails
# Usage: ./scripts/dvc_pull_smart.sh

echo "📥 Smart DVC Pull (Backblaze → DagsHub fallback)"
echo ""

echo "🔵 Attempting to pull from Backblaze (primary)..."
if dvc pull -r backblaze 2>/dev/null; then
    echo "✅ Successfully pulled from Backblaze"
    exit 0
else
    echo "⚠️  Backblaze pull failed (may have hit bandwidth limit)"
    echo ""
    echo "🟣 Falling back to DagsHub..."
    if dvc pull -r dagshub; then
        echo "✅ Successfully pulled from DagsHub"
        exit 0
    else
        echo "❌ Failed to pull from both remotes"
        echo "Please check your network connection and remote configurations"
        exit 1
    fi
fi
