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

# 2. Run FastV LLaVA
echo -e "\nRunning FastV LLaVA (K=288, R=3)..."
python run.py \
  --data COCO_VAL \
  --model llava_v1.5_7b_fastv_64 \
  --work-dir ./outputs_coco_64

echo ""
echo "================================================================"
echo "Evaluation completed!"
echo "FastV results: ./outputs_coco_64/llava_v1.5_7b_fastv_64/"
echo "================================================================"
