#!/bin/bash
# Hybrid (DyMU + FastV) Performance Benchmark Script
# Usage: ./run_hybrid_bench.sh [coco|mmbench] [num_samples]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set LD_LIBRARY_PATH for WSL (fix libcuda.so not found)
# export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

# Default dataset
DATASET="${1:-coco}"
NUM_SAMPLES="${2:-}"

# Set data file based on dataset
if [ "$DATASET" = "coco" ]; then
    DATA_FILE="perf_data/coco_val_500.json"
    REPORT_FILE="reports/hybrid_coco.json"
    LOG_FILE="logs/hybrid_coco.txt"
    MAX_NEW_TOKENS=64
    NUM_BEAMS=1
elif [ "$DATASET" = "mmbench" ]; then
    DATA_FILE="perf_data/mmbench_dev_en.json"
    REPORT_FILE="reports/hybrid_mmbench.json"
    LOG_FILE="logs/hybrid_mmbench.txt"
    MAX_NEW_TOKENS=128
    NUM_BEAMS=1
else
    echo "Unknown dataset: $DATASET. Use 'coco' or 'mmbench'"
    exit 1
fi

# Create output directories
mkdir -p reports logs

# Set PYTHONPATH for LLaVA, Transformers, and DyMU imports
export PYTHONPATH="${SCRIPT_DIR}/../dymu/src:${SCRIPT_DIR}/../FastV/src/transformers/src:${SCRIPT_DIR}/../FastV/src/LLaVA:${PYTHONPATH}"

# Build command
CMD="/home/aips/miniconda3/envs/vlm_hybrid/bin/python run_benchmark_hybrid.py \
    --data-file $DATA_FILE \
    --model-path liuhaotian/llava-v1.5-7b \
    --report-file $REPORT_FILE \
    --use-fastv \
    --fastv-k 72 \
    --fastv-r 3 \
    --r-total 504 \
    --threshold-path /home/aips/vlm/checkpoints/threshold_checkpoints/ViT-L-14-336-tome-72out.pth \
    --max-new-tokens $MAX_NEW_TOKENS \
    --num-beams $NUM_BEAMS \
    --temperature 0"

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num-samples $NUM_SAMPLES"
fi

echo "========================================"
echo "Hybrid Benchmark (DyMU + FastV)"
echo "Dataset: $DATASET"
echo "Data: $DATA_FILE"
echo "Report: $REPORT_FILE"
echo "Log: $LOG_FILE"
echo "========================================"

# Run benchmark and capture logs
$CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Report: $REPORT_FILE"
echo "Log: $LOG_FILE"
echo ""
echo "To analyze logs:"
echo "  python analyze_logs.py $LOG_FILE"
echo "========================================"
