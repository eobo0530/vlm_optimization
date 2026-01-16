
import pandas as pd
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

# Test with dummy data
gt_test = {
    '0': ['A child holding a flowered umbrella and petting a yak.', 'A young man holding an umbrella next to a herd of cattle.']
}
ref_matched = {
    '0': ['A child holding a umbrella']
}
ref_long = {
    '0': ['The image depicts a lively scene of a young boy feeding a cow in a field. The boy is holding an umbrella.']   
}

scorer = Cider()
print("Computing dummy scores...")

score_matched, _ = scorer.compute_score(gt_test, ref_matched)
print(f"Matched CIDEr: {score_matched}")

score_long, _ = scorer.compute_score(gt_test, ref_long)
print(f"Long CIDEr: {score_long}")


# Real data test
eval_file = '/home/user/vlm_opt_linux/VLMEvalKit/outputs/llava_v1.5_7b/llava_v1.5_7b_COCO_VAL.xlsx'
df = pd.read_excel(eval_file)
gt_real = {'0': eval(df.iloc[0]['answer'])}
ref_real = {'0': [str(df.iloc[0]['prediction'])]}

print("\nReal data 0th item:")
print(f"GT: {gt_real['0']}")
print(f"Ref: {ref_real['0']}")

score_real, _ = scorer.compute_score(gt_real, ref_real)
print(f"Real CIDEr: {score_real}")

bleu = Bleu(1)
score_bleu, _ = bleu.compute_score(gt_real, ref_real)
print(f"Real BLEU-1: {score_bleu}")
