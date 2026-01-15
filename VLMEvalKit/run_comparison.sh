#!/bin/bash

# FastV vs Baseline Comparison Script
# This script runs both baseline LLaVA and FastV-enabled LLaVA for comparison

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastv
cd /home/aips/vlm_optimization/VLMEvalKit

echo "=========================================="
echo "Running Baseline LLaVA (no FastV)"
echo "=========================================="
python run.py \
  --data MMBench_DEV_EN_V11 \
  --model llava_v1.5_7b \
  --work-dir ./outputs

echo ""
echo "=========================================="
echo "Running FastV-enabled LLaVA (k=100, r≈82.6%)"
echo "=========================================="
python run.py \
  --data MMBench_DEV_EN_V11 \
  --model llava_v1.5_7b_fastv \
  --work-dir ./outputs

echo ""
echo "=========================================="
echo "Both evaluations completed!"
echo "Compare results in ./outputs/"
echo "  - Baseline: ./outputs/llava_v1.5_7b/"
echo "  - FastV:    ./outputs/llava_v1.5_7b_fastv/"
echo "=========================================="
