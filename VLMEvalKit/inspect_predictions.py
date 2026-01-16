import pandas as pd
import os

file_path = '/home/user/vlm_opt_linux/VLMEvalKit/outputs/llava_v1.5_7b/llava_v1.5_7b_COCO_VAL.xlsx'
if os.path.exists(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"Total rows: {len(df)}")
        print("Columns:", df.columns.tolist())
        print("First 5 rows:")
        print(df.head()[['question', 'prediction']].to_string())
    except Exception as e:
        print(f"Error reading excel: {e}")
else:
    print("File not found.")
