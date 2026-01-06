#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")/.."

DATE=$(date +%Y-%m-%d)
OUTPUT_DIR="runs/baseline_${DATE}"
mkdir -p "$OUTPUT_DIR"

echo "Starting Baseline Benchmark for Qwen2-VL..."
echo "Output will be saved to ${OUTPUT_DIR}"

nsys profile --trace=cuda,nvtx --stats=true --force-overwrite true -o "${OUTPUT_DIR}/baseline_qwen" \
python microbench/benchmark_baseline.py --model_path Qwen/Qwen2-VL-7B-Instruct --output "${OUTPUT_DIR}/baseline_results_qwen.json" "$@"
