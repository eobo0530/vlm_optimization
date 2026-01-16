
import pickle
import pandas as pd
import sys
import os
from vlmeval.dataset import ImageCaptionDataset
from vlmeval.dataset.image_caption import COCO_Caption_Scorer

# Path to partial result
pkl_path = '/home/user/vlm_opt_linux/VLMEvalKit/outputs/llava_v1.5_7b/T20260115_Gecf16da3/01_COCO_VAL.pkl'

print(f"Loading partial results from {pkl_path}...")
with open(pkl_path, 'rb') as f:
    preds = pickle.load(f)

# Load GT
print("Loading COCO_VAL dataset...")
dataset = ImageCaptionDataset(dataset='COCO_VAL')
# Dataset loading usually puts data in dataset.data
# dataset.data is a pandas DataFrame

# Align predictions and GT
ref = {}
gt = {}

count = 0
for idx, prediction in preds.items():
    # idx is likely 0-based or 1-based depending on pkl saving. 
    # Usually in vlmeval it's 0-based index of the dataset.
    # The pkl keys are integers.
    
    # Ensure prediction is string
    if not isinstance(prediction, str):
        continue
        
    # Keys are 1-based, iloc is 0-based
    line = dataset.data.iloc[idx - 1]
    
    # Eval needs dict of list of strings
    ref[str(idx)] = [prediction]
    
    # GT answer is typically a list of strings stored in 'answer' column
    # The 'answer' column in dataset might be a string representation of list or list
    ans = line['answer']
    if isinstance(ans, str):
        ans = eval(ans)
    gt[str(idx)] = ans
    count += 1

print(f"Evaluate on {count} samples...")

if count > 0:
    scorer = COCO_Caption_Scorer(ref, gt)
    scores = scorer.compute_scores()
    print("Scores:", scores)
else:
    print("No samples to evaluate.")
