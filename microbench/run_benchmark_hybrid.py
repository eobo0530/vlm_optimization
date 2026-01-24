#!/usr/bin/env python3
"""
Hybrid (DyMU + FastV) Performance Benchmark Script
Measures: E2E latency, TTFT, TBT, TPS, VRAM
Outputs profiling logs for analyze_logs.py
"""

import os
import json
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import LogitsProcessor
import open_clip

# Set Profiling Env Var
os.environ["ENABLE_PROFILING"] = "1"

# LLaVA modules (requires PYTHONPATH set to FastV/src/LLaVA)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def find_thresholds(threshold_finding_checkpoint):
    """Load DyMU learned thresholds from checkpoint."""
    print(f"Loading thresholds from {threshold_finding_checkpoint}")
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


class PerformanceMetrics:
    """Collects and reports performance metrics."""
    
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.e2e_latencies = []
        self.ttfts = []  # Time To First Token
        self.tbts = []   # Time Between Tokens
        self.gpu_mem_peaks = []
        self.output_lens = []
        self.input_lens = []

    def update(self, e2e, ttft, tbt_avg, gpu_mem, out_len, in_len):
        self.e2e_latencies.append(e2e)
        if ttft is not None: 
            self.ttfts.append(ttft)
        if tbt_avg is not None: 
            self.tbts.append(tbt_avg)
        self.gpu_mem_peaks.append(gpu_mem)
        self.output_lens.append(out_len)
        self.input_lens.append(in_len)
    
    def report(self):
        def p50(x): return np.percentile(x, 50) if x else 0
        def p95(x): return np.percentile(x, 95) if x else 0
        def avg(x): return np.mean(x) if x else 0
        
        stats = {
            "e2e_p50": p50(self.e2e_latencies),
            "e2e_p95": p95(self.e2e_latencies),
            "ttft_p50": p50(self.ttfts),
            "ttft_p95": p95(self.ttfts),
            "ttft_avg": avg(self.ttfts),
            "tps_p50": p50([1.0/t if t > 0 else 0 for t in self.tbts]),
            "tps_avg": avg([1.0/t if t > 0 else 0 for t in self.tbts]),
            "tbt_p50": p50(self.tbts),
            "tbt_p95": p95(self.tbts),
            "tbt_avg": avg(self.tbts),
            "vram_max_mb": max(self.gpu_mem_peaks) if self.gpu_mem_peaks else 0,
            "avg_output_len": avg(self.output_lens),
            "avg_input_len": avg(self.input_lens)
        }

        print(f"\n===== Performance Report (Hybrid: DyMU + FastV) =====")
        if self.metadata:
            print(f"Dataset: {self.metadata.get('dataset', 'Unknown')}")
        print(f"Total Samples: {len(self.e2e_latencies)}")
        print(f"E2E Latency (s): P50={stats['e2e_p50']:.4f}, P95={stats['e2e_p95']:.4f}")
        print(f"TTFT (s):        P50={stats['ttft_p50']:.4f}, P95={stats['ttft_p95']:.4f}, Avg={stats['ttft_avg']:.4f}")
        print(f"TBT (s):         P50={stats['tbt_p50']:.4f}, P95={stats['tbt_p95']:.4f}, Avg={stats['tbt_avg']:.4f}")
        print(f"Decode TPS:      P50={stats['tps_p50']:.2f}, Avg={stats['tps_avg']:.2f}")
        print(f"Peak VRAM (MB):  Max={stats['vram_max_mb']:.2f}")
        print(f"Avg Output Len:  {stats['avg_output_len']:.1f} tokens")
        print(f"Avg Input Len:   {stats['avg_input_len']:.1f} tokens")
        print("=" * 50)
        return stats

    def save_json(self, filename):
        stats = self.report()
        full_record = {
            "meta": self.metadata,
            "summary": stats,
            "samples": []
        }
        for i in range(len(self.e2e_latencies)):
            full_record["samples"].append({
                "e2e": self.e2e_latencies[i],
                "ttft": self.ttfts[i] if i < len(self.ttfts) else None,
                "tbt": self.tbts[i] if i < len(self.tbts) else None,
                "vram": self.gpu_mem_peaks[i],
                "out_len": self.output_lens[i],
                "in_len": self.input_lens[i]
            })
            
        with open(filename, 'w') as f:
            json.dump(full_record, f, indent=4)
        print(f"Report saved to {filename}")


class TimingLogitsProcessor(LogitsProcessor):
    """Records timing for each generated token."""
    
    def __init__(self, start_time):
        self.start_time = start_time
        self.first_token_time = None
        self.token_times = []
        
    def __call__(self, input_ids, scores):
        now = time.time()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)
        return scores


def main():
    parser = argparse.ArgumentParser(description="Hybrid (DyMU + FastV) Benchmark")
    parser.add_argument("--data-file", type=str, required=True, help="Path to benchmark data JSON")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0, help="0 for greedy")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--report-file", type=str, default=None, help="Path to save JSON report")
    
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search size")
    
    # DyMU parameters
    parser.add_argument("--r-total", type=int, default=504, help="DyMU r_total (tokens to merge)")
    parser.add_argument("--threshold-path", type=str, 
                        default="../dymu/checkpoints/threshold_checkpoints/ViT-L-14-336-tome-72out.pth",
                        help="Path to DyMU threshold checkpoint")
    
    # FastV parameters
    parser.add_argument("--use-fastv", action="store_true", default=True, help="Enable FastV")
    parser.add_argument("--fastv-k", type=int, default=288, help="FastV attention rank")
    parser.add_argument("--fastv-r", type=int, default=3, help="FastV aggregation layer")
    parser.add_argument("--fastv-sys-length", type=int, default=35, help="FastV system prompt length")
    
    args = parser.parse_args()

    # === 1. Setup DyMU Configuration ===
    threshold_checkpoint_path = args.threshold_path
    if not os.path.isabs(threshold_checkpoint_path):
        # Make relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        threshold_checkpoint_path = os.path.join(script_dir, args.threshold_path)
    
    learned_thresholds = find_thresholds(threshold_checkpoint_path)
    print(f"Loaded DyMU thresholds: {len(learned_thresholds)} layers")
    
    # DyMU config for token merging
    tome_kwargs = {
        'pretrained': "openai",
        'pretrained_origin_tag': "openai",
        'merge_mode': "batch_level",
        "repeat_merged_tokens": False,
        "r_total": args.r_total,
        "specified_thresholds": learned_thresholds
    }

    # === 2. Load Model ===
    print(f"Loading model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name + "_tome_openai",  # Signals ToMe vision tower
        tome_kwargs=tome_kwargs,
        device_map="cuda",
        torch_dtype=torch.float16
    )

    # [Optimization Fix] Force replace vision tower with ToMe version if not loaded
    # because FastV builder checks config string for "tome" which is missing in standard config.
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTowerToMe
    vision_tower = model.get_vision_tower()
    
    if not isinstance(vision_tower, CLIPVisionTowerToMe):
        print(f"⚠️  Vanilla Vision Tower detected ({type(vision_tower).__name__}). Swapping to CLIPVisionTowerToMe for DyMU...")
        
        vision_tower_name = vision_tower.vision_tower_name
        if "openai/clip-vit-large-patch14-336" in vision_tower_name:
            vision_tower_name = "ViT-L-14-336-tome-72out"
            
        # Create new ToMe tower
        # CLIPVisionTowerToMe.__init__ does not accept kwargs, so we pass kwargs to load_model
        new_tower = CLIPVisionTowerToMe(vision_tower_name, args=model.config, delay_load=True)
        new_tower.load_model(device="cuda", dtype=model.dtype, **tome_kwargs)
        
        # Replace in model (LlavaLlamaModel -> vision_tower)
        model.model.vision_tower = new_tower
        
        # Update image_processor
        image_processor = new_tower.image_processor
        print(f"✅ DyMU Vision Tower injected. Thresholds: {len(args.threshold_path) > 0}")

    # === 3. Setup FastV (applied after DyMU token merging) ===
    if args.use_fastv:
        # Calculate reduced image token length after DyMU merging (576 -> 72)
        original_tokens = 576
        merged_token_count = original_tokens - args.r_total  # 576 - 504 = 72
        
        model.config.use_fast_v = True
        model.config.fast_v_inplace = True
        model.config.fast_v_sys_length = args.fastv_sys_length
        model.config.fast_v_image_token_length = 576 # Set to max to cover dynamic DyMU output (96-152 tokens)
        model.config.fast_v_attention_rank = args.fastv_k
        model.config.fast_v_agg_layer = args.fastv_r
        model.config.output_attentions = True  # Required for FastV
        
        # Propagate config to model instance
        if hasattr(model.model, 'reset_fastv'):
            model.model.reset_fastv()
        
        print(f"🚀 FastV enabled: K={args.fastv_k}, R={args.fastv_r}, image_len={merged_token_count}")
    else:
        print("FastV disabled")

    # === 4. Load Data ===
    with open(args.data_file, 'r') as f:
        file_content = json.load(f)
    
    if isinstance(file_content, dict) and 'data' in file_content:
        data_items = file_content['data']
        metadata = file_content.get('meta', {})
    else:
        data_items = file_content
        metadata = {}
    
    print(f"Loaded {len(data_items)} items from {args.data_file}")
    
    if args.num_samples and args.num_samples > 0:
        data_items = data_items[:args.num_samples]
        print(f"Limiting to {len(data_items)} samples")

    metrics = PerformanceMetrics(metadata)
    conv_mode = "llava_v1"

    # === 5. Run Benchmark ===
    pbar = tqdm(data_items, desc="Hybrid Benchmark")
    for i, item in enumerate(pbar):
        torch.cuda.reset_peak_memory_stats()
        
        # Prepare Prompt & Images
        prompt_struct = item['prompt_struct']
        images = []
        curr_text = ""
        
        for p in prompt_struct:
            if p['type'] == 'image':
                img_path = p['value']
                
                # Handle LMUData path
                if "LMUData" in img_path:
                    # Use current user's home directory instead of hardcoded /home/user
                    user_home = os.path.expanduser("~")
                    img_path = img_path.replace("../../LMUData", os.path.join(user_home, "LMUData"))
                
                # Handle relative paths
                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(img_path)
                
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                    curr_text += DEFAULT_IMAGE_TOKEN + "\n"
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    
            elif p['type'] == 'text':
                curr_text += p['value']
        
        curr_text = curr_text.strip()

        # Apply LLaVA conversation template
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], curr_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process images
        if images:
            image_tensor = process_images(images, image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        else:
            image_tensor = None
            
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_len = input_ids.shape[1]

        # [Profiling] Measure Pure Vision Latency
        if image_tensor is not None:
            torch.cuda.synchronize()
            v_start = time.time()
            with torch.inference_mode():
                # This calls vision_tower + projector
                _ = model.get_model().get_vision_tower()(image_tensor)
            torch.cuda.synchronize()
            v_end = time.time()
            print(f"[Profiling] VisionStep: {(v_end - v_start) * 1000:.4f} ms")

        # Generation with timing
        torch.cuda.synchronize()
        t_start = time.time()
        timing_processor = TimingLogitsProcessor(t_start)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                max_new_tokens=args.max_new_tokens,
                logits_processor=[timing_processor],
                output_attentions=args.use_fastv,  # For FastV
                num_beams=args.num_beams  # Use argument
            )
            
        torch.cuda.synchronize()
        t_end = time.time()

        # [Profiling Output for analyze_logs.py]
        if timing_processor.first_token_time:
            # We map TTFT (Vision + Prefill + 1st Token) to "VisionStep" for compatibility
            ttft_ms = (timing_processor.first_token_time - t_start) * 1000
            print(f"[Profiling] TTFT (Vision+Prefill): {ttft_ms:.4f} ms")

            # Output per-token latency
            current_t = timing_processor.first_token_time
            if hasattr(timing_processor, 'token_times') and timing_processor.token_times:
                for t in timing_processor.token_times:
                    if t <= current_t: continue
                    step_ms = (t - current_t) * 1000
                    print(f"[Profiling] LLM_Decode: {step_ms:.4f} ms")
                    current_t = t

        # Calculate metrics
        e2e = t_end - t_start
        
        if output_ids.shape[1] < input_len:
            output_len = output_ids.shape[1]
        else:
            output_len = output_ids.shape[1] - input_len
        
        ttft = 0
        if timing_processor.first_token_time:
            ttft = timing_processor.first_token_time - t_start
        
        tbt_avg = 0
        if output_len > 1 and len(timing_processor.token_times) > 1:
            generate_time = timing_processor.token_times[-1] - timing_processor.first_token_time
            tbt_avg = generate_time / (output_len - 1)
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        metrics.update(e2e, ttft, tbt_avg, peak_mem, output_len, input_len)
        pbar.set_postfix({'E2E': f"{e2e:.2f}s", 'TPS': f"{1/tbt_avg if tbt_avg > 0 else 0:.1f}"})

    # Print compression stats if available
    try:
        if hasattr(model, 'get_vision_tower'):
            vision_tower = model.get_vision_tower()
            if hasattr(vision_tower, 'print_compression_stats'):
                print("\n[Profiling] Checking compression stats...")
                vision_tower.print_compression_stats()
    except Exception as e:
        print(f"\n[Warning] Could not print compression stats: {e}")

    # Output results
    if args.report_file:
        metrics.save_json(args.report_file)
    else:
        metrics.report()


if __name__ == "__main__":
    main()
