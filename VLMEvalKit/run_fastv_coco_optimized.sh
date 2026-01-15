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
  --data COCO_VAL \
  --model llava_v1.5_7b_fastv \
  --work-dir ./outputs_opt \
  --reuse-aux 0

echo ""
echo "==================================="
echo "Evaluation completed!"
echo "Results saved to: ./outputs_opt/llava_v1.5_7b_fastv/"
echo "==================================="
