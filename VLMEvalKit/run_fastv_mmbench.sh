#!/bin/bash

# FastV MMBench Evaluation Script
# This script runs FastV-enabled LLaVA on MMBench benchmark

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fastv

# Navigate to VLMEvalKit directory
cd /home/aips/vlm_optimization/VLMEvalKit

# Run evaluation
python run.py \
  --data MMBench_DEV_EN \
  --model llava_v1.5_7b_fastv \
  --work-dir ./outputs \
  --reuse

echo ""
echo "==================================="
echo "Evaluation completed!"
echo "Results saved to: ./outputs/llava_v1.5_7b_fastv/"
echo "==================================="
