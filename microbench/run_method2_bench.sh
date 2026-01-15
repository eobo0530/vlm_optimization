#!/bin/bash

# Configuration
MODEL_PATH="liuhaotian/llava-v1.5-7b"
IMAGE_PATH="/home/aips/LMUData/images/COCO/2418.jpg"
PROMPT="Describe this image in detail."
MAX_TOKENS=256

# Check if image exists, if not use a default from coco
if [ ! -f "$IMAGE_PATH" ]; then
    IMAGE_PATH="/home/aips/vlm_optimization/VLMEvalKit/data/coco/val2017/000000039769.jpg"
fi

echo "================================================================"
echo "   FASTV METHOD 2 (STATIC KV-CACHE PRUNING)マイクロベンチマーク   "
echo "================================================================"

# 1. Baseline
echo -e "\n[1/3] Running Baseline LLaVA..."
python /home/aips/vlm_optimization/microbench/benchmark_method2.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --max-new-tokens $MAX_TOKENS > baseline_m2.log 2>&1

# 2. FastV K=288 (R=50%)
echo -e "\n[2/3] Running FastV Static (K=288, R=3)..."
python /home/aips/vlm_optimization/microbench/benchmark_method2.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --max-new-tokens $MAX_TOKENS \
    --use-fastv --k 288 --r 3 > fastv_k288_m2.log 2>&1

# 3. FastV K=144 (R=25%)
echo -e "\n[3/3] Running FastV Static (K=144, R=3)..."
python /home/aips/vlm_optimization/microbench/benchmark_method2.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --max-new-tokens $MAX_TOKENS \
    --use-fastv --k 144 --r 3 > fastv_k144_m2.log 2>&1

# Summary Table
echo -e "\n\n=========================================================================================="
echo -e "   CONFIG          |  TTFT (ms) |  TBT (ms)  |  TPS       |  Peak VRAM (MB) | E2E (s)"
echo -e "------------------------------------------------------------------------------------------"

grep -E "TTFT|TBT|TPS|Peak VRAM|E2E" baseline_m2.log | awk -F': ' '{printf "%-18s", $2}' | xargs echo -e "Baseline        |"
grep -E "TTFT|TBT|TPS|Peak VRAM|E2E" fastv_k288_m2.log | awk -F': ' '{printf "%-18s", $2}' | xargs echo -e "FastV (K=288)   |"
grep -E "TTFT|TBT|TPS|Peak VRAM|E2E" fastv_k144_m2.log | awk -F': ' '{printf "%-18s", $2}' | xargs echo -e "FastV (K=144)   |"

echo -e "=========================================================================================="
echo -e "Note: TBT (Time Between Tokens) is the key metric for Decoding speedup."
