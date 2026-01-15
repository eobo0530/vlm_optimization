#!/bin/bash

# FastV Evaluation with VLMEvalKit
# This script runs FastV-enabled LLaVA on MMBench and COCO using VLMEvalKit

# Configuration
MODEL_PATH="liuhaotian/llava-v1.5-7b"
WORK_DIR="./outputs_fastv"

# FastV parameters
USE_FAST_V=true
FAST_V_SYS_LENGTH=35
FAST_V_IMAGE_TOKEN_LENGTH=576
FAST_V_ATTENTION_RANK=100  # Adjust this to control pruning ratio
FAST_V_AGG_LAYER=3

# Datasets to evaluate
DATASETS="MMBench_DEV_EN_V11 COCO_VAL"

# Run evaluation with FastV enabled
python run_fastv_vlmeval.py \
    --model-path ${MODEL_PATH} \
    --use-fast-v \
    --fast-v-sys-length ${FAST_V_SYS_LENGTH} \
    --fast-v-image-token-length ${FAST_V_IMAGE_TOKEN_LENGTH} \
    --fast-v-attention-rank ${FAST_V_ATTENTION_RANK} \
    --fast-v-agg-layer ${FAST_V_AGG_LAYER} \
    --data ${DATASETS} \
    --work-dir ${WORK_DIR} \
    --mode all \
    --verbose

# Alternative: Run using config file
# python run_fastv_vlmeval.py --config config_fastv_example.json
