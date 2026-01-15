
import json
import os
import subprocess
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../VLMEvalKit'))
import numpy as np
import pandas as pd
from vlmeval.dataset import build_dataset
from vlmeval.smp import dump

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def normalize_path(path, root_dir):
    try:
        if path and os.path.exists(path):
            return os.path.relpath(path, root_dir)
    except:
        pass
    return path # Return original if fails

def extract_prompt_info(prompt_struct, root_dir):
    image_paths = []
    text_prompt = ""
    
    for p in prompt_struct:
        if p['type'] == 'image':
            # Normalize path
            abs_path = p['value']
            rel_path = normalize_path(abs_path, root_dir)
            p['value'] = rel_path # Update struct for portability
            image_paths.append(rel_path)
            
        elif p['type'] == 'text':
            text_prompt += p['value']
            
    return prompt_struct, image_paths, text_prompt

def prepare_coco_subset(output_file, seed=42, sample_size=500):
    print(f"Preparing COCO subset (seed={seed}, size={sample_size})...")
    dataset_name = 'COCO_VAL'
    dataset = build_dataset(dataset_name)
    data = dataset.data
    
    # Filter or Sample
    np.random.seed(seed)
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        # Sort indices to ensure consistent order
        indices = np.sort(indices)
        subset = data.iloc[indices].copy()
    else:
        indices = np.arange(len(data))
        subset = data.copy()
    
    # Root dir for relative paths (Assuming run from VLMEvalKit root)
    root_dir = os.getcwd()
    git_hash = get_git_revision_short_hash()
    
    meta = {
        'dataset': dataset_name,
        'seed': seed,
        'sample_size': len(subset),
        'vlmeval_git': git_hash,
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    items = []
    for i in range(len(subset)):
        line = subset.iloc[i]
        prompt_struct = dataset.build_prompt(line)
        
        # Extract and normalize
        prompt_struct, image_paths, text_prompt = extract_prompt_info(prompt_struct, root_dir)
        
        item = {
            'index': int(line['index']) if 'index' in line else i,
            'image_id': int(line['index']) if 'index' in line else i, # COCO uses index as id often, strictly check if image_id exists
            'dataset': dataset_name,
            'image_paths': image_paths,
            'text_prompt': text_prompt,
            'prompt_struct': prompt_struct,
        }
        # If COCO has specific image_id in columns
        if 'image_id' in line:
             item['image_id'] = int(line['image_id'])

        items.append(item)
        
    final_output = {'meta': meta, 'data': items}
    dump(final_output, output_file)
    print(f"Saved {len(items)} items to {output_file} (Relative paths from {root_dir})")
    return final_output

def prepare_mmbench(output_file):
    print("Preparing MMBench DEV EN (Full)...")
    dataset_name = 'MMBench_DEV_EN'
    dataset = build_dataset(dataset_name)
    data = dataset.data
    
    root_dir = os.getcwd()
    git_hash = get_git_revision_short_hash()
    
    meta = {
        'dataset': dataset_name,
        'sample_size': len(data),
        'vlmeval_git': git_hash,
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    items = []
    for i in range(len(data)):
        line = data.iloc[i]
        prompt_struct = dataset.build_prompt(line)
        
        # Extract and normalize
        prompt_struct, image_paths, text_prompt = extract_prompt_info(prompt_struct, root_dir)

        item = {
            'index': int(line['index']) if 'index' in line else i,
            'dataset': dataset_name,
            'image_paths': image_paths,
            'text_prompt': text_prompt,
            'prompt_struct': prompt_struct,
        }
        items.append(item)
        
    final_output = {'meta': meta, 'data': items}
    dump(final_output, output_file)
    print(f"Saved {len(items)} items to {output_file}")
    return final_output

if __name__ == "__main__":
    os.makedirs('perf_data', exist_ok=True)
    # Ensure we are in microbench directory
    if not os.path.basename(os.getcwd()) == 'microbench':
         print("Warning: Please run this script from the microbench directory.")
    
    prepare_coco_subset('perf_data/coco_val_500.json')
    prepare_mmbench('perf_data/mmbench_dev_en.json')