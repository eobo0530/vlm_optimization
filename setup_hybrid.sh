#!/bin/bash

# Hybrid LLaVA (DyMU + FastV) Environment Setup Script
# This script installs all local components in editable mode.

set -e

echo "Starting Hybrid VLM Environment Setup..."

# 1. Install external dependencies
if [ -f "requirements_dist.txt" ]; then
    echo "Installing external dependencies from requirements_dist.txt..."
    pip install -r requirements_dist.txt
fi

# 2. Install Patched Transformers (v4.31.0 with FastV patches)
echo "Installing patched transformers in editable mode..."
cd FastV/src/transformers
pip install -e .
cd ../../..

# 3. Install Patched LLaVA (with Hybrid DyMU patches)
echo "Installing patched LLaVA in editable mode..."
cd FastV/src/LLaVA
pip install -e .
cd ../../..

# 4. Install DyMU Package
echo "Installing DyMU package in editable mode..."
cd dymu
pip install -e .
cd ..

# 5. Install VLMEvalKit
echo "Installing VLMEvalKit in editable mode..."
cd VLMEvalKit
pip install -e .
cd ..

echo "Hybrid VLM Environment Setup Complete!"
echo "You can now run benchmarks using VLMEvalKit."
