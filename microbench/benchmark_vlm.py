import argparse
import time
import torch
import os
import sys
import json
from PIL import Image
import numpy as np

# Add FastV transformers and LLaVA to path
FASTV_TRANSFORMERS_PATH = '/home/aips/FastV/src/transformers/src'
LLAVA_PATH = '/home/aips/FastV/src/LLaVA'

for path in [FASTV_TRANSFORMERS_PATH, LLAVA_PATH]:
    if os.path.exists(path):
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added {path} to sys.path")

from transformers import AutoTokenizer, AutoConfig
# We need the specific LLaVA loading logic. 
# Reusing the logic from the previously viewed fastv_llava.py
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    print("Error: LLaVA library not found. Please ensure it's installed.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Micro-benchmark for LLaVA and FastV")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b", help="Model path")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the benchmark image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Benchmark prompt")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    
    # FastV Parameters
    parser.add_argument("--use-fastv", action="store_true", help="Enable FastV")
    parser.add_argument("--k", type=int, default=100, help="FastV K value (attention rank)")
    parser.add_argument("--r", type=int, default=3, help="FastV R value (aggregation layer)")
    parser.add_argument("--inplace", action="store_true", help="Use FastV inplace mode")
    
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarking")
    
    return parser.parse_args()

def benchmark_model(args):
    device = "cuda"
    model_name = get_model_name_from_path(args.model_path)
    
    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto"
    )
    
    if args.use_fastv:
        print(f"Configuring FastV: K={args.k}, R={args.r}, Inplace={args.inplace}")
        model.config.use_fast_v = True
        model.config.fast_v_inplace = args.inplace
        model.config.fast_v_sys_length = 35 # Default for LLaVA v1.5
        model.config.fast_v_image_token_length = 576
        model.config.fast_v_attention_rank = args.k
        model.config.fast_v_agg_layer = args.r
        # Re-init fastv state if available
        if hasattr(model.model, 'reset_fastv'):
            model.model.reset_fastv()

    # Prepare input
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = process_images([image] * args.batch_size, image_processor, model.config).to(device, dtype=torch.float16)
    
    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{args.prompt} ASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    # Replicate input_ids for batch size
    input_ids = input_ids.repeat(args.batch_size, 1)
    
    print(f"Starting warmup (Batch Size: {args.batch_size})...")
    for _ in range(2):
        with torch.inference_mode():
            model.generate(input_ids, images=image_tensor, max_new_tokens=10, do_sample=False)
    
    print(f"Running benchmark (target: {args.max_new_tokens} tokens per sequence)...")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Timing Events
    start_event = torch.cuda.Event(enable_timing=True)
    first_token_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Custom generate loop to capture TTFT and TBT
    with torch.inference_mode():
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Prefill + First Token
        output = model.generate(
            input_ids, 
            images=image_tensor, 
            max_new_tokens=1, 
            do_sample=False, 
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=args.use_fastv # Required for FastV to work in prefill
        )
        torch.cuda.synchronize()
        first_token_time = time.perf_counter()
        
        ttft = (first_token_time - start_time) * 1000 # ms
        
        # Decoding phase
        next_token_ids = output.sequences
        past_key_values = getattr(output, 'past_key_values', None)
        
        if past_key_values is None:
            print("Warning: past_key_values not found in generate() output. Re-calculating for decode phase...")
            # Fallback for older transformers/llava versions
            with torch.inference_mode():
                full_out = model(
                    input_ids, 
                    images=image_tensor, 
                    use_cache=True,
                    output_attentions=args.use_fastv
                )
                past_key_values = full_out.past_key_values

        decode_start_time = time.perf_counter()
        num_decode_tokens = args.max_new_tokens - 1
        
        # Optimization: We can just use generate with max_new_tokens for the rest
        # but to get TPS accurately for the decode phase:
        final_output = model.generate(
            next_token_ids,
            images=image_tensor,
            past_key_values=past_key_values,
            max_new_tokens=num_decode_tokens,
            do_sample=False,
            use_cache=True,
            output_attentions=args.use_fastv # Keep it consistent
        )
        torch.cuda.synchronize()
        decode_end_time = time.perf_counter()
        
        total_decode_time = decode_end_time - decode_start_time
        # TPS is (num tokens * batch size) / time
        tps = (num_decode_tokens * args.batch_size) / total_decode_time if total_decode_time > 0 else 0
        tbt = (total_decode_time * 1000) / num_decode_tokens if num_decode_tokens > 0 else 0
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
        e2e_latency = (decode_end_time - start_time)
        
    # Results
    metrics = {
        "Model": args.model_path,
        "FastV_Enabled": args.use_fastv,
        "K": args.k if args.use_fastv else "N/A",
        "R": args.r if args.use_fastv else "N/A",
        "E2E Latency (s)": round(e2e_latency, 3),
        "TTFT (ms)": round(ttft, 2),
        "TBT (ms/tok)": round(tbt, 2),
        "TPS (tokens/s)": round(tps, 2),
        "Peak VRAM (MB)": round(peak_vram, 2),
        "Generated Tokens": args.max_new_tokens
    }
    
    print("\n" + "="*30)
    print(" BENCHMARK RESULTS ")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k:18}: {v}")
    print("="*30)
    
    return metrics

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    benchmark_model(args)
