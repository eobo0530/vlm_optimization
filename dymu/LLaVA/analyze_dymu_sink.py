
import argparse
import torch
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_sink_scores(model, tokenizer, image_processor, args):
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
            max_new_tokens=1, 
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True
        )

    # image token indices
    token_ids = input_ids[0]
    image_token_indices = (token_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    
    if len(image_token_indices) == 0:
        print("Warning: No image tokens found in input.")
        return None

    img_start = image_token_indices[0].item()
    
    num_layers = model.config.num_hidden_layers
    sink_ratios = []
    
    for layer_idx in range(num_layers):
        attn = output_ids.attentions[0][layer_idx] # (1, 32, q_len, kv_len)
        attn = attn[0, :, -1, :] # (num_heads, kv_len)
        
        # Ratio of attention directed at tokens BEFORE image tokens (system/user prompt prefix)
        prompt_attn = attn[:, 0:img_start].sum(dim=-1) # (num_heads,)
        total_attn = attn.sum(dim=-1) # (num_heads,)
        
        avg_sink = (prompt_attn / total_attn).mean().item()
        sink_ratios.append(avg_sink)
        
    return sink_ratios

def plot_sink_analysis(baseline_sinks, dymu_sinks, savedir):
    os.makedirs(savedir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    layers = list(range(len(baseline_sinks)))
    plt.plot(layers, baseline_sinks, marker='o', label='Baseline (Llama-1.5-7B)', color='blue')
    plt.plot(layers, dymu_sinks, marker='s', label='DyMU (r=504)', color='orange')
    
    # FastV 80% Sink Threshold
    plt.axhline(y=0.8, color='red', linestyle='--', label='80% Attention Sink Threshold')
    
    plt.xlabel("LLM Layer Index")
    plt.ylabel("Attention Ratio to System Prompt")
    plt.title("System Prompt Attention Sink Analysis (Baseline vs DyMU)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(savedir, "attention_sink_comparison.png"))
    plt.close()
    
    # Save raw data
    data = {
        "baseline_sinks": baseline_sinks,
        "dymu_sinks": dymu_sinks
    }
    with open(os.path.join(savedir, "sink_metrics.json"), "w") as f:
        json.dump(data, f, indent=4)
    print(f"Sink analysis graphics and metrics saved in {savedir}")

def force_activate_dymu(model, tome_kwargs):
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTowerToMe
    import open_clip
    
    vision_tower = model.get_vision_tower()
    tower_name = vision_tower.vision_tower_name
    if "openai" in tower_name and "336" in tower_name:
         tower_name = "ViT-L-14-336"
         tome_kwargs['pretrained'] = "openai"
         tome_kwargs['pretrained_origin_tag'] = "openai"
         
    try:
        base_config = open_clip.get_model_config(tower_name)
        if base_config:
            vision_cfg = base_config.get('vision_cfg', {})
            tome_cfg = {
                'r_total': tome_kwargs.get('r_total', 0),
                'merge_mode': tome_kwargs.get('merge_mode', 'batch_level'),
                'r_schedule': 'constant'
            }
            vision_cfg['tome_cfg'] = tome_cfg
            tome_kwargs['vision_cfg'] = vision_cfg
    except:
        pass

    new_tower = CLIPVisionTowerToMe(tower_name, args=model.config, delay_load=True)
    new_tower.load_model(device=model.device, dtype=model.dtype, **tome_kwargs)
    model.model.vision_tower = new_tower
    return new_tower

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image-file", type=str, default="images/001.jpg")
    parser.add_argument("--query", type=str, default="Describe the image.")
    args = parser.parse_args()
    
    args.conv_mode = "llava_v1"
    args.device = "cuda"

    # 1. DyMU
    print("=== Analyzing DyMU Attention Sink ===")
    tome_kwargs_dymu = {
        'pretrained': "openai",
        'merge_mode': "batch_level",
        "repeat_merged_tokens": False, # Use repo's native unmerging
        "r_total": 504,
    }
    
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_v1.5_7b", 
        device=args.device, tome_kwargs=tome_kwargs_dymu 
    )
    force_activate_dymu(model, tome_kwargs_dymu)
    
    # Verification Print
    vision_tower = model.get_vision_tower()
    test_image = load_image(args.image_file)
    test_tensor = process_images([test_image], image_processor, model.config).to(model.device, dtype=model.dtype)
    with torch.no_grad():
        v_features, padding_mask, size, pos_tracking = vision_tower(test_tensor)
        # v_features is [1, 72, 1024] 
        print(f"VERIFICATION: DyMU Merged Tokens: {v_features.shape[1]}")
        if v_features.shape[1] > 100:
             print("WARNING: DyMU merging count is high. Expected ~72.")

    dymu_sinks = get_sink_scores(model, tokenizer, image_processor, args)
    
    del model
    torch.cuda.empty_cache()
    
    # 2. Baseline
    print("=== Analyzing Baseline Attention Sink ===")
    tome_kwargs_baseline = {
        'pretrained': "openai",
        'merge_mode': "batch_level",
        "repeat_merged_tokens": False, 
        "r_total": 0,
    }
    
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_v1.5_7b", 
        device=args.device, tome_kwargs=tome_kwargs_baseline
    )
    force_activate_dymu(model, tome_kwargs_baseline)
    baseline_sinks = get_sink_scores(model, tokenizer, image_processor, args)
    
    # Compare
    if dymu_sinks and baseline_sinks:
        plot_sink_analysis(baseline_sinks, dymu_sinks, savedir="sink_analysis_results")

if __name__ == "__main__":
    main()
