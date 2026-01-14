#!/bin/bash
set -e

# Directory for profiling outputs
PROFILE_DIR="profiling_outputs"
mkdir -p $PROFILE_DIR

# 1. Profile Baseline
echo "Running Baseline Profiling..."
nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --output="${PROFILE_DIR}/baseline_profile" \
    --force-overwrite=true \
    python3 /home/user/vlm_opt_linux/microbench/benchmark_perf.py \
    --data-file /home/user/vlm_opt_linux/microbench/perf_data/coco_val_500.json \
    --report-file "${PROFILE_DIR}/baseline_report_coco.json"

echo "Running Baseline Profiling (MMBench)..."
nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --output="${PROFILE_DIR}/baseline_profile_mmbench" \
    --force-overwrite=true \
    python3 /home/user/vlm_opt_linux/microbench/benchmark_perf.py \
    --data-file /home/user/vlm_opt_linux/microbench/perf_data/mmbench_dev_en.json \
    --report-file "${PROFILE_DIR}/baseline_report_mmbench.json"

# 2. Profile DyMU (Commented out until baseline is done or requested)
# echo "Running DyMU Profiling..."
# ...

echo "Profiling Complete. Results saved in ${PROFILE_DIR}/"
ls -lh $PROFILE_DIR
