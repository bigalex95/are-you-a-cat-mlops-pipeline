#!/bin/bash
# Sync DVC data across all remotes
# Usage: ./scripts/dvc_sync_remotes.sh

set -e

echo "🔄 Syncing DVC data across all remotes..."
echo ""

# Push to Backblaze
echo "📤 Pushing to Backblaze..."
if dvc push -r backblaze; then
    echo "✅ Successfully pushed to Backblaze"
else
    echo "❌ Failed to push to Backblaze"
    exit 1
fi

echo ""

# Push to DagsHub
echo "📤 Pushing to DagsHub..."
if dvc push -r dagshub; then
    echo "✅ Successfully pushed to DagsHub"
else
    echo "❌ Failed to push to DagsHub"
    exit 1
fi

echo ""
echo "🎉 Sync complete! Data is now available on all remotes."