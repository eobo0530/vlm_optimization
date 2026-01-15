#!/bin/bash

# Configuration
MODEL_PATH="liuhaotian/llava-v1.5-7b"
IMAGE_PATH=$1
if [ -z "$IMAGE_PATH" ]; then
    IMAGE_PATH=$(find /home/aips/LMUData/images -name "*.jpg" | head -n 1)
fi

if [ -z "$IMAGE_PATH" ]; then
    echo "Error: No image provided and no images found in /home/aips/LMUData/images"
    exit 1
fi

echo "=============================================="
echo "      ADVANCED MICRO-BENCHMARK: FASTV        "
echo "=============================================="
echo "Image: $IMAGE_PATH"

# Function to run and extract
run_bench() {
    local name=$1
    local params=$2
    echo -e "\n>>> Running $name..."
    python benchmark_vlm.py --model-path $MODEL_PATH --image-path "$IMAGE_PATH" $params --max-new-tokens 100 > "${name}.bench"
}

# 1. Baseline Batch 1
run_bench "Baseline_B1" ""

# 2. FastV K100 No-Inplace Batch 1
run_bench "FastV_K100_B1" "--use-fastv --k 100"

# 3. FastV K100 Inplace Batch 1
run_bench "FastV_K100_Inplace_B1" "--use-fastv --k 100 --inplace"

# 4. Baseline Batch 4
run_bench "Baseline_B4" "--batch-size 4"

# 5. FastV K100 Inplace Batch 4
run_bench "FastV_K100_Inplace_B4" "--use-fastv --k 100 --inplace --batch-size 4"

echo -e "\n=============================================================================="
echo "                        SUMMARY OF ADVANCED RESULTS                          "
echo "=============================================================================="

function extract_val() {
    grep "$1" "$2" | awk -F': ' '{print $2}'
}

printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "Config" "TTFT(ms)" "TPS(t/s)" "VRAM(MB)" "E2E(s)"
echo "------------------------------------------------------------------------------"

configs=("Baseline_B1" "FastV_K100_B1" "FastV_K100_Inplace_B1" "Baseline_B4" "FastV_K100_Inplace_B4")

for c in "${configs[@]}"; do
    f="${c}.bench"
    ttft=$(extract_val "TTFT (ms)" "$f")
    tps=$(extract_val "TPS (tokens/s)" "$f")
    vram=$(extract_val "Peak VRAM (MB)" "$f")
    e2e=$(extract_val "E2E Latency (s)" "$f")
    printf "%-25s | %-10s | %-10s | %-10s | %-10s\n" "$c" "$ttft" "$tps" "$vram" "$e2e"
done

echo "=============================================================================="
echo "Observation: Inplace mode and larger batch sizes showcase FastV's efficiency."
