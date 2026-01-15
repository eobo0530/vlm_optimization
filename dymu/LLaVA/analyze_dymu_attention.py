
import argparse
import torch
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

# Import from local LLaVA
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import open_clip

def find_thresholds(threshold_finding_checkpoint):
    print(f"Loading threshold finding checkpoint from {threshold_finding_checkpoint}...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-336-tome-72out", 
        pretrained=threshold_finding_checkpoint
    )
    tome_vision_encoder = model.visual.trunk if hasattr(model.visual, "trunk") else model.visual
    blocks = tome_vision_encoder.blocks if hasattr(tome_vision_encoder, "blocks") else tome_vision_encoder.transformer.resblocks
    learned_thresholds = []
    for i, block in enumerate(blocks):
        learned_thresholds.append(block.threshold.item())
    return learned_thresholds

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_attention_scores(model, tokenizer, image_processor, args, dymu_enabled=True):
    # Construct prompt
    prompt = args.query
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Load image
    image = load_image(args.image_file)
    image_tensor = process_images([image], image_processor, args)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=1, # We only need the attention of the first generated token
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True
        )

    # Extract attention
    # output_ids.attentions is a tuple (one element per generated token)
    # Each element is a tuple (one element per layer)
    # Each layer is a tensor (batch, num_heads, seq_len, seq_len)
    
    # We want attention of the *first new token* attending to the *image tokens*.
    # seq_len includes system prompt + image tokens + user prompt + new token
    
    # Let's get the attentions from the last layer (or average across layers?)
    # FastV usually looks across layers or specific layers. Let's start with Layer 15 (middle-ish) or similar.
    # LLaVA-1.5-7B has 32 layers.
    
    # Structure: attentions[generation_step][layer] -> (batch, heads, q_len, kv_len)
    # Since use_cache=True, q_len=1 for generation steps.
    
    # image token indices in input_ids?
    # We simplified: The prompt has image tokens. We need to find where they are.
    # input_ids shape: [1, seq_len]
    # Image tokens are where input_ids == IMAGE_TOKEN_INDEX (-200)
    
    token_ids = input_ids[0]
    image_token_indices = (token_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    
    if len(image_token_indices) == 0:
        print("Warning: No image tokens found in input.")
        return None

    img_start = image_token_indices[0].item()
    num_visual_tokens = 576 # Standard for LLaVA-1.5 336
    img_end = img_start + num_visual_tokens
    print(f"Expect {num_visual_tokens} visual tokens at indices {img_start}:{img_end}")

    layer_attentions = []
    target_layers = [1, 10, 20, 30] 
    
    for layer_idx in target_layers:
        attn = output_ids.attentions[0][layer_idx] # (1, 32, q_len, kv_len)
        
        # In prefill step, q_len is the whole sequence length. 
        # But we want the attention of the LAST token (the generated one or the last prompt token if output_attentions=True).
        # Actually generate returns attentions for each step. 
        # step 0: prefill. step 1: first token generation.
        # Wait, if max_new_tokens=1, attentions[0] might be the prefill attentions.
        
        # Let's check the kv_len
        kv_len = attn.shape[-1]
        print(f"Layer {layer_idx} attention shape: {attn.shape}")
        
        # If we just generated 1 token, the last token in prefill (index -1) 
        # is the one that predicts the first new token.
        
        attn = attn[0, :, -1, :] # (num_heads, kv_len)
        
        # Slice out image part
        image_attn = attn[:, img_start:img_end]
        
        # Average over heads
        avg_attn = image_attn.mean(dim=0) # (num_tokens,)
        
        layer_attentions.append(avg_attn.cpu().numpy())
        
    return layer_attentions

def plot_and_compare(baseline_attns, dymu_attns, savedir):
    os.makedirs(savedir, exist_ok=True)
    
    metrics = []
    
    target_layers = [1, 10, 20, 30]
    
    for i, layer_idx in enumerate(target_layers):
        b_attn = baseline_attns[i]
        d_attn = dymu_attns[i]
        
        # distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(b_attn, color='blue', label='Baseline', stat="density", kde=True, alpha=0.4)
        sns.histplot(d_attn, color='orange', label='DyMU', stat="density", kde=True, alpha=0.4)
        plt.title(f"Attention Score Distribution (Layer {layer_idx})")
        plt.legend()
        plt.savefig(os.path.join(savedir, f"dist_layer_{layer_idx}.png"))
        plt.close()
        
        # Kendall's Tau
        tau, p_value = kendalltau(b_attn, d_attn)
        
        # Top-K Overlap (Simple Intersection)
        k = len(b_attn) // 2 # 50%
        top_k_b = np.argsort(b_attn)[-k:]
        top_k_d = np.argsort(d_attn)[-k:]
        
        overlap = len(set(top_k_b).intersection(set(top_k_d)))
        iou = overlap / (len(set(top_k_b).union(set(top_k_d))))
        overlap_ratio = overlap / k
        
        metrics.append({
            "layer": layer_idx,
            "kendall_tau": tau,
            "top_50_overlap": overlap_ratio,
             "iou": iou
        })
        
        print(f"Layer {layer_idx}: Kendall Tau={tau:.4f}, Top-50% Overlap={overlap_ratio:.4f}")

    # Save metrics
    with open(os.path.join(savedir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def find_thresholds(threshold_finding_checkpoint):
    if not os.path.exists(threshold_finding_checkpoint):
        print(f"Warning: Checkpoint {threshold_finding_checkpoint} not found. Using default/random thresholds.")
        return None
        
    print(f"Loading threshold finding checkpoint from {threshold_finding_checkpoint}...")
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336-tome-72out", 
            pretrained=threshold_finding_checkpoint
        )
        tome_vision_encoder = model.visual.trunk if hasattr(model.visual, "trunk") else model.visual
        blocks = tome_vision_encoder.blocks if hasattr(tome_vision_encoder, "blocks") else tome_vision_encoder.transformer.resblocks
        learned_thresholds = []
        for i, block in enumerate(blocks):
            learned_thresholds.append(block.threshold.item())
        return learned_thresholds
    except Exception as e:
        print(f"Error loading thresholds: {e}. detailed: using none")
        return None

def force_activate_dymu(model, tome_kwargs):
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTowerToMe
    import open_clip
    
    vision_tower = model.get_vision_tower()
    
    # We always force swap if we want DyMU enabled/configured properly with specific kwargs
    # because the original loaded tower is likely standard CLIP.
    
    print("Forcing CLIPVisionTowerToMe setup...")
    tower_name = vision_tower.vision_tower_name
    if "openai" in tower_name and "336" in tower_name:
         tower_name = "ViT-L-14-336"
         tome_kwargs['pretrained'] = "openai"
         tome_kwargs['pretrained_origin_tag'] = "openai"
         
    # Fetch base config and inject tome_cfg
    print(f"Fetching base config for {tower_name} to inject ToMe params...")
    try:
        base_config = open_clip.get_model_config(tower_name)
        if base_config:
            vision_cfg = base_config.get('vision_cfg', {})
            # Inject tome_cfg
            tome_cfg = {
                'r_total': tome_kwargs.get('r_total', 0),
                'merge_mode': tome_kwargs.get('merge_mode', 'batch_level'),
                'r_schedule': 'constant'
            }
            vision_cfg['tome_cfg'] = tome_cfg
            
            # Pass this modified vision_cfg to load_model
            # Note: create_model merges kwargs into model_cfg. passing 'vision_cfg' overrides it.
            # We must ensure vision_cfg has all original keys.
            tome_kwargs['vision_cfg'] = vision_cfg
    except Exception as e:
        print(f"Error fetching/patching config: {e}. Proceeding without patch.")

    new_tower = CLIPVisionTowerToMe(tower_name, args=model.config, delay_load=True)
    new_tower.load_model(device=model.device, dtype=model.dtype, **tome_kwargs)
    
    # Replace
    model.model.vision_tower = new_tower
    
    return new_tower

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image-file", type=str, default="images/001.jpg")
    parser.add_argument("--query", type=str, default="Describe the image.")
    parser.add_argument("--threshold-checkpoint", type=str, default="checkpoints/threshold_checkpoints/ViT-L-14-336-tome-72out.pth")
    args = parser.parse_args()
    
    # Common Args
    args.model_name = "llava_v1.5_tome_openai"
    args.conv_mode = "llava_v1"
    args.sep = ","
    args.temperature = 0
    args.top_p = None
    args.num_beams = 1
    args.max_new_tokens = 512
    args.device = "cuda"
    args.load_8bit = False
    args.load_4bit = False
    
    # 1. Run DyMU
    print("=== Running DyMU Config ===")
    learned_thresholds = find_thresholds(args.threshold_checkpoint)
    tome_kwargs_dymu = {
        'pretrained': "openai",
        'merge_mode': "batch_level",
        "repeat_merged_tokens": True, # Ensure 576 tokens for LLM comparison
        "r_total": 504,
    }
    if learned_thresholds is not None:
        tome_kwargs_dymu["specified_thresholds"] = learned_thresholds
    
    args.tome_kwargs = tome_kwargs_dymu
    
    # Force reload
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        None, 
        args.model_name, 
        load_8bit=args.load_8bit, 
        load_4bit=args.load_4bit, 
        device=args.device,
        tome_kwargs=tome_kwargs_dymu 
    )
    
    # Swapping logic to ensure DyMU is active
    vision_tower = force_activate_dymu(model, tome_kwargs_dymu)
    image_processor = vision_tower.image_processor # Update processor
    
    # Verification Print
    test_image = load_image(args.image_file)
    test_tensor = process_images([test_image], image_processor, model.config).to(model.device, dtype=model.dtype)
    with torch.no_grad():
        v_features, _, _, _ = vision_tower(test_tensor)
        # Move to CPU for safer unique calculation
        num_unique = torch.unique(v_features[0].cpu(), dim=0).shape[0]
        print(f"VERIFICATION: DyMU Visual Shape: {v_features.shape[1]}, Unique Tokens (Information Content): {num_unique}")
        if num_unique > 100:
             print("WARNING: DyMU merging count is high. Expected ~72-93.")

    dymu_attns = get_attention_scores(model, tokenizer, image_processor, args, dymu_enabled=True)
    
    del model
    torch.cuda.empty_cache()
    
    # 2. Run Baseline (DyMU disabled)
    print("=== Running Baseline Config ===")
    # Disable merging by setting r_total = 0
    tome_kwargs_baseline = {
        'pretrained': "openai",
        'pretrained_origin_tag': "openai",
        'merge_mode': "batch_level", # Matters less if r=0
        "repeat_merged_tokens": False, 
        "r_total": 0,
    }
    
    args.tome_kwargs = tome_kwargs_baseline
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        None, 
        args.model_name, 
        load_8bit=args.load_8bit, 
        load_4bit=args.load_4bit, 
        device=args.device,
        tome_kwargs=tome_kwargs_baseline
    )

    # Note: Baseline needs to use ToMe tower with r=0? 
    # Or strict baseline uses original CLIP?
    # Using ToMe tower with r=0 is "functionally" baseline.
    # We should swap here too to ensure consistent processing pipeline.
    vision_tower = force_activate_dymu(model, tome_kwargs_baseline)
    image_processor = vision_tower.image_processor

    baseline_attns = get_attention_scores(model, tokenizer, image_processor, args, dymu_enabled=False)
    
    # Compare
    if dymu_attns and baseline_attns:
        plot_and_compare(baseline_attns, dymu_attns, savedir="attention_analysis_results")
        print("Analysis complete. Results saved in 'attention_analysis_results'.")
    else:
        print("Error: Could not extract attentions.")

if __name__ == "__main__":
    main()
