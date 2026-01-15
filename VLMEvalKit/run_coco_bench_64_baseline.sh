#!/bin/bash

# FastV COCO Evaluation Script (Max 64 Tokens)
# This script runs both baseline LLaVA and FastV-enabled LLaVA on COCO_VAL

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastv

# Navigate to VLMEvalKit directory
cd /home/aips/vlm_optimization/VLMEvalKit

echo "================================================================"
echo "Starting COCO Benchmarking (max_new_tokens=64)"
echo "================================================================"

# 1. Run Baseline LLaVA
echo -e "\nRunning Baseline LLaVA (v1.5 7B)..."
python run.py \
  --data COCO_VAL \
  --model llava_v1.5_7b_64 \
  --work-dir ./outputs_coco_64

echo ""
echo "================================================================"
echo "Evaluation completed!"
echo "Baseline results: ./outputs_coco_64/llava_v1.5_7b_64/"
echo "================================================================"
