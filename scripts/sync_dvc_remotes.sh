#!/bin/bash

# sync_dvc_remotes.sh
# This script syncs DVC data across multiple remotes: Backblaze and DagsHub.

set -e  # Exit immediately if a command exits with a non-zero status

# Function to sync DVC data
sync_dvc() {
    local remote_name=$1
    echo "Starting sync with remote: $remote_name"
    
    if dvc pull -r "$remote_name"; then
        echo "Successfully synced with remote: $remote_name"
    else
        echo "Error syncing with remote: $remote_name" >&2
        exit 1
    fi
}

# Sync with Backblaze
sync_dvc "backblaze"

# Sync with DagsHub
sync_dvc "dagsHub"

echo "All remotes synced successfully!"