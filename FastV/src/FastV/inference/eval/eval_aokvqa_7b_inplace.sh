#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/home/aips/FastV/src/LLaVA:/home/aips/FastV/src

# LLaVA v1.5 7B Model Path
model_path=/home/aips/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234
output_path=aokvqa_eval_7b_inplace_fp16
mkdir -p $output_path

# Dataset Path Verification
dataset_path="/home/aips/FastV/data/aokvqa/validation"
if [ ! -d "$dataset_path" ]; then
    echo "❌ Error: Dataset directory $dataset_path not found!"
    exit 1
fi

echo "✅ Dataset found. Starting benchmarks..."

# rank equals to (1-R)*N_Image_Tokens, N_Image_Tokens = 576
# R=25% Dropped -> rank=432
# R=50% Dropped -> rank=288
# R=75% Dropped -> rank=144
rank_list=(144 288 432) 
Ks=(2) 

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
        echo "--------------------------------------------------"
        echo "🚀 Running FastV Inplace: Rank $rank, Layer $k (FP16)"
        /home/aips/miniconda3/envs/fastv/bin/python /home/aips/FastV/src/FastV/inference/eval/inference_aokvqa.py \
            --model-path "$model_path" \
            --use-fast-v \
            --fast-v-inplace \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length 576 \
            --fast-v-attention-rank "$rank" \
            --fast-v-agg-layer "$k" \
            --limit-samples 10 \
            --output-path "$output_path/aokvqa_7b_inplace_rank${rank}_layer${k}.json"
    done
done

# Baseline (7B Vanilla)
echo "--------------------------------------------------"
echo "🚀 Running Baseline (Vanilla 7B FP16)"
/home/aips/miniconda3/envs/fastv/bin/python /home/aips/FastV/src/FastV/inference/eval/inference_aokvqa.py \
    --model-path "$model_path" \
    --limit-samples 10 \
    --output-path "$output_path/aokvqa_7b_baseline.json"

echo "--------------------------------------------------"
echo "✅ All benchmarks completed!"
