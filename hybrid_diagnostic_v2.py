import os
import sys
import torch
from PIL import Image
from types import MethodType

# Ensure paths are correct
DYMU_SRC = '/home/aips/vlm/dymu/src'
FASTV_TRANSFORMERS = '/home/aips/vlm/FastV/src/transformers/src'
FASTV_LLAVA = '/home/aips/vlm/FastV/src/LLaVA'

for path in [DYMU_SRC, FASTV_TRANSFORMERS, FASTV_LLAVA]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTowerToMe

# Import our HybridLLaVA but simulate its logic here for direct inspection
def diagnostic():
    model_path = "liuhaotian/llava-v1.5-7b"
    image_path = "/home/aips/vlm/VLMEvalKit/assets/apple.jpg"
    
    print("\n" + "="*60)
    print(" Hybrid Logical Mapping Diagnostic (v2) ".center(60, "="))
    print("="*60)

    import llava
    print(f"[*] Using LLaVA from: {llava.__file__}")
    
    # 1. DyMU Config (Logical Mapping ON)
    # Note: HybridLLaVA sets repeat_merged_tokens=False internally now for logical mapping
    
    # We need to add the VLMEvalKit path to import HybridLLaVA
    VLMEVAL_PATH = '/home/aips/vlm/VLMEvalKit'
    if os.path.exists(VLMEVAL_PATH) and VLMEVAL_PATH not in sys.path:
        sys.path.insert(0, VLMEVAL_PATH)
    
    from vlmeval.vlm.hybrid_llava import HybridLLaVA
    
    print(f"[*] Initializing HybridLLaVA...")
    # HybridLLaVA(model_path, **kwargs)
    # It will automatically hot-swap the vision tower
    hybrid_model = HybridLLaVA(
        model_path=model_path,
        fast_v_attention_rank=36,
        r_total=504
    )
    
    model = hybrid_model.model
    tokenizer = hybrid_model.tokenizer
    image_processor = hybrid_model.image_processor

    print("\n[ Analysis 1: Sequence Lengths ]")
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config).to("cuda", dtype=torch.float16)
    
    prompt = "A chat between a curious human and an artificial intelligence assistant. USER: <image>\nDescribe this image. ASSISTANT:"
    
    # We want to see the position_ids from prepare_inputs_labels_for_multimodal
    print("[*] Running multimodal preparation...")
    # Monkey patch forward to catch inputs
    old_forward = model.model.forward
    captured_inputs = {}
    
    def forward_hook(self, *args, **kwargs):
        captured_inputs['position_ids'] = kwargs.get('position_ids')
        captured_inputs['input_ids'] = kwargs.get('input_ids')
        captured_inputs['inputs_embeds'] = kwargs.get('inputs_embeds')
        return old_forward(*args, **kwargs)
    
    model.model.forward = forward_hook.__get__(model.model, type(model.model))

    with torch.inference_mode():
        model.generate(
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda"),
            images=image_tensor,
            max_new_tokens=5
        )

    pos_ids = captured_inputs['position_ids']
    embeds = captured_inputs['inputs_embeds']
    
    print(f"[V] Physical Sequence Length: {embeds.shape[1]}")
    print(f"[V] Max Position ID: {pos_ids.max().item()}")
    
    if embeds.shape[1] < pos_ids.max().item():
        print("[SUCCESS] Physical length is smaller than max Position ID. Logical mapping is working!")
    else:
        print("[FAILURE] Physical length matches Max Position ID. Tokens are still being replicated.")

    print("\n[ Analysis 2: Position ID Continuity ]")
    # Check for jumps in position_ids
    diffs = pos_ids[0, 1:] - pos_ids[0, :-1]
    jumps = (diffs > 1).nonzero()
    if len(jumps) > 0:
        print(f"[V] Found {len(jumps)} jumps in position_ids.")
        for idx in jumps[:3]:
            i = idx.item()
            print(f"    - Jump at index {i}: {pos_ids[0, i].item()} -> {pos_ids[0, i+1].item()}")
    else:
        print("[X] No jumps found in position_ids. Spatial context might be lost.")

    print("\n" + "="*60)

if __name__ == "__main__":
    diagnostic()
