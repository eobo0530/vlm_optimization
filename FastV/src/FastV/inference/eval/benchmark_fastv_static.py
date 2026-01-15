import os
import sys
import json
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import argparse

# Add relevant paths
sys.path.insert(0, "/home/aips/FastV/src/transformers/src")
sys.path.append("/home/aips/FastV/src/FastV/lmms-eval")

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from transformers.models.llama.modeling_llama import LlamaModel
except ImportError:
    print("Error: LLaVA modules not found. Please ensure the path is correct.")
    exit(1)

# Import the forward method from the custom file
from fastv_kvcache import FastVLlamaModel

def patch_model_with_fastv(model, k, ratio):
    """
    Patches the model to use Static Pruning (KV Cache) logic from fastv_kvcache.py
    """
    model.config.fast_v_k = k
    model.config.fast_v_ratio = ratio
    model.config.fast_v_sys_length = 35
    model.config.fast_v_image_token_length = 576
    
    # Set the attribute that the patched forward expects
    model.model.last_attention = None
    
    # Bind the method
    model.model.forward = FastVLlamaModel.forward.__get__(model.model, LlamaModel)
    
    print(f"🚀 Model patched with Static Pruning (KV Cache) logic. K={k}, Ratio={ratio}")

def main(args):
    # Reset peak memory BEFORE loading the model to capture EVERYTHING
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Load model
    model_name = get_model_name_from_path(args.model_path)
    # Toggle 4-bit vs FP16
    load_4bit = not args.load_in_fp16
    print(f"Loading model in {'FP16' if args.load_in_fp16 else '4-bit'} mode...")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, load_8bit=False, load_4bit=load_4bit, device="cuda:0"
    )

    if args.load_in_fp16:
        print("Ensuring model and vision tower are on GPU...")
        model.to(device='cuda', dtype=torch.float16)

    # Verify model state
    print(f"🔍 Model Dtype: {model.dtype}")
    print(f"🔍 Model Device: {model.device}")
    
    # Check if any layer is on CPU
    cpu_layers = [i for i, l in enumerate(model.model.layers) if l.self_attn.q_proj.weight.device.type == 'cpu']
    if cpu_layers:
        print(f"⚠️ Warning: Layers {cpu_layers} are on CPU!")
    else:
        print("✅ All layers are on GPU.")

    if image_processor is None:
        print("Warning: image_processor is None. Attempting to load from vision tower...")
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor

    if args.use_fast_v:
        patch_model_with_fastv(model, args.k, args.ratio)

    # Benchmark settings
    num_warmup = 1
    num_runs = args.num_runs
    
    # Prepare input
    image = Image.new('RGB', (224, 224), color = (73, 109, 137))
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    prompt = "Describe this image in detail."
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_full = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_full, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    
    print(f"Benchmarking with {input_ids.shape[1]} input tokens...")
    
    # Force everything to CUDA
    input_ids = input_ids.to('cuda')
    image_tensor = image_tensor.to('cuda', dtype=torch.float16)
    model.to('cuda')
    
    prefill_latencies = []
    decode_latencies = [] # per token
    e2e_latencies = []
    peak_vrams = []
    num_generated_tokens = []

    for i in range(num_warmup + num_runs):
        if i >= num_warmup:
            print(f"Run {i - num_warmup + 1}/{num_runs}...")
        
        torch.cuda.synchronize()
        
        # Generation with timing
        with torch.inference_mode():
            # Prefill
            t0 = time.time()
            outputs = model(
                input_ids,
                images=image_tensor,
                use_cache=True,
                output_attentions=True,
                return_dict=True
            )
            torch.cuda.synchronize()
            t1 = time.time()
            prefill_latency = t1 - t0
            
            past_key_values = outputs.past_key_values
            next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
            # Decoding
            t2 = time.time()
            generated_token_ids = []
            curr_input_ids = next_token_ids
            
            # FORCE generating exactly args.max_new_tokens (Ignore EOS for fair benchmark)
            for _ in range(args.max_new_tokens - 1):
                outputs = model(
                    curr_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
                curr_input_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                generated_token_ids.append(curr_input_ids)
                # Removed EOS check for fair benchmark
            
            torch.cuda.synchronize()
            t3 = time.time()
            decode_time = t3 - t2
            tokens_count = len(generated_token_ids) + 1
            avg_decode_latency = decode_time / tokens_count
            e2e_latency = t3 - t0
            peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3) # GB
            
            if i >= num_warmup:
                prefill_latencies.append(prefill_latency)
                decode_latencies.append(avg_decode_latency)
                e2e_latencies.append(e2e_latency)
                peak_vrams.append(peak_vram)
                num_generated_tokens.append(tokens_count)

    avg_prefill = sum(prefill_latencies) / len(prefill_latencies)
    avg_decode = sum(decode_latencies) / len(decode_latencies)
    avg_e2e = sum(e2e_latencies) / len(e2e_latencies)
    avg_vram = sum(peak_vrams) / len(peak_vrams)
    avg_tokens = sum(num_generated_tokens) / len(num_generated_tokens)
    avg_tps = 1.0 / avg_decode if avg_decode > 0 else 0
    
    print("\n" + "="*35)
    print(f"Results (FastV Static={args.use_fast_v})")
    print(f"Avg Prefill Latency: {avg_prefill:.4f} s")
    print(f"Avg Decode Latency:  {avg_decode:.4f} s/token")
    print(f"Avg Decode TPS:      {avg_tps:.2f} tokens/s")
    print(f"Avg E2E Latency:     {avg_e2e:.4f} s")
    print(f"Avg Peak VRAM:       {avg_vram:.2f} GB")
    print(f"Avg Tokens Generated: {avg_tokens:.1f}")
    print("="*35)
    
    result = {
        "use_fast_v": args.use_fast_v,
        "k": args.k,
        "ratio": args.ratio,
        "prefill_latency": avg_prefill,
        "decode_latency": avg_decode,
        "tps": avg_tps,
        "e2e_latency": avg_e2e,
        "peak_vram_gb": avg_vram,
        "avg_tokens": avg_tokens,
        "runs": num_runs
    }
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--use-fast-v", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--load-in-fp16", action="store_true", help="Load model in FP16 instead of 4-bit")
    args = parser.parse_args()
    main(args)
