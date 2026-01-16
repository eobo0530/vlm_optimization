import pickle
import pandas as pd

file_path = '/home/user/vlm_opt_linux/VLMEvalKit/outputs/llava_v1.5_7b/T20260115_Gecf16da3/01_COCO_VAL.pkl'
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        # Usually it's a dict mapping index to prediction result?
        first_key = next(iter(data))
        print(f"Example Item ({first_key}): {data[first_key]}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        print(f"First Item: {data[0]}")
except Exception as e:
    print(f"Error: {e}")
