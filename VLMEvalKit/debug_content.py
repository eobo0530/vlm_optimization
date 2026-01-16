
import pickle
from vlmeval.dataset import ImageCaptionDataset

pkl_path = '/home/user/vlm_opt_linux/VLMEvalKit/outputs/llava_v1.5_7b/T20260115_Gecf16da3/01_COCO_VAL.pkl'
with open(pkl_path, 'rb') as f:
    preds = pickle.load(f)

dataset = ImageCaptionDataset(dataset='COCO_VAL')

print(f"Total samples: {len(preds)}")
for i, (idx, pred) in enumerate(preds.items()):
    if i >= 5: break
    line = dataset.data.iloc[idx]
    gt = line['answer']
    print(f"=== Sample {idx} ===")
    print(f"Prediction ({len(pred.split())} words):")
    print(pred)
    print(f"Ground Truth:")
    print(gt)
    print("-" * 20)
