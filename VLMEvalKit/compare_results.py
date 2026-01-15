
import pandas as pd

# Paths to the result files
low_score_path = "/home/aips/vlm_optimization/VLMEvalKit/outputs_coco_64/llava_v1.5_7b_fastv_64/T20260112_G7f546e48/llava_v1.5_7b_fastv_64_COCO_VAL.xlsx"
high_score_path = "/home/aips/vlm_optimization/VLMEvalKit/outputs_opt/llava_v1.5_7b_fastv/T20260110_G7f546e48/llava_v1.5_7b_fastv_COCO_VAL.xlsx"

try:
    df_low = pd.read_excel(low_score_path)
    df_high = pd.read_excel(high_score_path)

    print("=== LOW SCORE SAMPLE (Score: 15) ===")
    print(df_low.iloc[0]['question'])
    print("\n" + "="*50 + "\n")

    print("=== HIGH SCORE SAMPLE (Score: 79) ===")
    print(df_high.iloc[0]['question'])
    
    print("\n" + "="*50 + "\n")
    print("Column names in LOW:", df_low.columns)
    print("Column names in HIGH:", df_high.columns)

except Exception as e:
    print(f"Error reading excel files: {e}")
