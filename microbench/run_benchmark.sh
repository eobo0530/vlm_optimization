#!/bin/bash

# Configuration
MODEL_PATH="liuhaotian/llava-v1.5-7b"
# Use the first image found in LMUData/images if available, else require input
IMAGE_PATH=$1
if [ -z "$IMAGE_PATH" ]; then
    IMAGE_PATH=$(find /home/aips/LMUData/images -name "*.jpg" | head -n 1)
fi

if [ -z "$IMAGE_PATH" ]; then
    echo "Error: No image provided and no images found in /home/aips/LMUData/images"
    exit 1
fi

echo "Using image: $IMAGE_PATH"
echo "Starting Micro-benchmarks..."

# 1. Baseline
echo -e "\n>>> [1/3] Running Baseline..."
python benchmark_vlm.py --model-path $MODEL_PATH --image-path "$IMAGE_PATH" --max-new-tokens 100 > baseline.bench

# 2. FastV K=288
echo -e "\n>>> [2/3] Running FastV (K=288, R=3)..."
python benchmark_vlm.py --model-path $MODEL_PATH --image-path "$IMAGE_PATH" --use-fastv --k 288 --r 3 --max-new-tokens 100 > fastv_k288.bench

# 3. FastV K=100
echo -e "\n>>> [3/3] Running FastV (K=100, R=3)..."
python benchmark_vlm.py --model-path $MODEL_PATH --image-path "$IMAGE_PATH" --use-fastv --k 100 --r 3 --max-new-tokens 100 > fastv_k100.bench

echo -e "\n=============================================="
echo "         SUMMARY OF BENCHMARK RESULTS         "
echo "=============================================="

function extract_val() {
    grep "$1" "$2" | awk -F': ' '{print $2}'
}

printf "%-20s | %-12s | %-12s | %-12s\n" "Metric" "Baseline" "FastV(K288)" "FastV(K100)"
echo "------------------------------------------------------------------------------"

metrics=("TTFT (ms)" "TPS (tokens/s)" "Peak VRAM (MB)" "E2E Latency (s)")

for m in "${metrics[@]}"; do
    v1=$(extract_val "$m" baseline.bench)
    v2=$(extract_val "$m" fastv_k288.bench)
    v3=$(extract_val "$m" fastv_k100.bench)
    printf "%-20s | %-12s | %-12s | %-12s\n" "$m" "$v1" "$v2" "$v3"
done

echo "=============================================="
