#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "Please run './install_mac.sh' first."
    exit 1
fi

echo "[Launcher] Activating environment..."
source .venv/bin/activate

# Set OpenBLAS thread safety
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "[Launcher] Starting App (Hot Reload Mode)..."
python reloader.py