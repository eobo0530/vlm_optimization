
import pandas as pd
import os

xlsx_path = "/home/user/vlm_opt_linux/VLMEvalKit/outputs/llava_v1.5_7b/T20260114_G08aeab2d/llava_v1.5_7b_COCO_VAL.xlsx"

try:
    df = pd.read_excel(xlsx_path)
    print(f"Total samples: {len(df)}")
    print("\n--- Sample Predictions ---")
    # Print distinct predictions to see variety
    print(df['prediction'].sample(5).values)
    
    print("\n--- Check for 'Error' or Empty ---")
    errors = df[df['prediction'].astype(str).str.contains("Error|Fail", case=False)]
    print(f"Error count: {len(errors)}")
except Exception as e:
    print(f"Failed to read xlsx: {e}")
