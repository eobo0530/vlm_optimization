import json
import pandas as pd
import numpy as np
import os
from collections import Counter
import string

# Configs
baseline_dir = "/home/user/vlm_opt_linux/VLMEvalKit/outputs/llava_v1.5_7b"
dymu_dir = "/home/user/vlm_opt_linux/dymu/outputs/llava_v1.5_7b_dymu"

# Helper to find latest file
def get_latest_file(base_dir, pattern):
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for f in filenames:
            if pattern in f and f.endswith('.xlsx'):
                files.append(os.path.join(root, f))
    if not files:
        return None
    return sorted(files, key=os.path.getmtime)[-1]

baseline_file = get_latest_file(baseline_dir, "COCO_VAL")
dymu_file = get_latest_file(dymu_dir, "COCO_VAL")

print(f"Baseline File: {baseline_file}")
print(f"DyMU File: {dymu_file}")

if not baseline_file or not dymu_file:
    print("One or both result files not found.")
    exit(1)

# Load Data
df_base = pd.read_excel(baseline_file)
df_dymu = pd.read_excel(dymu_file)

# 1. Matching Logic Check
print("\n--- 1. Matching Logic Check ---")
common_ids = set(df_base['index']) & set(df_dymu['index'])
print(f"Baseline samples: {len(df_base)}")
print(f"DyMU samples: {len(df_dymu)}")
print(f"Common samples: {len(common_ids)}")

if len(df_base) != len(df_dymu) or len(common_ids) != len(df_base):
    print("WARNING: Mismatch in sample counts or IDs!")

# Sort by index to ensure alignment
df_base = df_base.set_index('index').loc[list(common_ids)].reset_index()
df_dymu = df_dymu.set_index('index').loc[list(common_ids)].reset_index()

# 2. Caption Statistics
print("\n--- 2. Caption Statistics ---")
def get_stats(text_series):
    lengths = text_series.astype(str).apply(lambda x: len(x.split()))
    return {
        'mean_len': lengths.mean(),
        'median_len': lengths.median(),
        'std_len': lengths.std()
    }

base_stats = get_stats(df_base['prediction'])
dymu_stats = get_stats(df_dymu['prediction'])

print("Baseline Lengths (words):", base_stats)
print("DyMU Lengths (words):    ", dymu_stats)

# Noun Analysis (Simple heuristic: words starting with capital or simple existence)
# Note: Real POS tagging needs nltk/spacy, using simple approach here to avoid deps
def simple_noun_ratio(text):
    words = text.split()
    if not words: return 0
    # Heuristic: roughly count object-like words (longer than 3 chars, not stop words)
    # This is a weak proxy but works for "concreteness" check
    return len([w for w in words if len(w) > 3]) / len(words)

df_base['noun_proxy'] = df_base['prediction'].astype(str).apply(simple_noun_ratio)
df_dymu['noun_proxy'] = df_dymu['prediction'].astype(str).apply(simple_noun_ratio)

print(f"Baseline 'Solidity' Ratio: {df_base['noun_proxy'].mean():.4f}")
print(f"DyMU 'Solidity' Ratio:     {df_dymu['noun_proxy'].mean():.4f}")

# 3. Post-processing Check
print("\n--- 3. Post-processing Check ---")
def check_prefixes(series):
    prefixes = ["The image", "A photo of", "There is", "Assistant:"]
    counts = {p: series.astype(str).str.startswith(p).sum() for p in prefixes}
    return counts

print("Baseline Prefixes:", check_prefixes(df_base['prediction']))
print("DyMU Prefixes:    ", check_prefixes(df_dymu['prediction']))

# 4. Qualitative Comparison (Random Samples)
print("\n--- 4. Sample Comparison (Random 5) ---")
samples = df_base.sample(5)
for idx, row in samples.iterrows():
    img_id = row['index']
    base_pred = row['prediction']
    dymu_pred = df_dymu[df_dymu['index'] == img_id]['prediction'].values[0]
    
    print(f"\n[Image ID: {img_id}]")
    print(f"Base: {base_pred}")
    print(f"DyMU: {dymu_pred}")
