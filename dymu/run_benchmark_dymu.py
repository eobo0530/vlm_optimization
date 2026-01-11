
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

# LLaVA modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates

def find_thresholds(threshold_finding_checkpoint):
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

# Metric Collector (Copied from microbench/benchmark_perf.py)
class PerformanceMetrics:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.e2e_latencies = []
        self.ttfts = [] 
        self.tbts = []  
        self.gpu_mem_peaks = []
        self.output_lens = []
        self.input_lens = []

    def update(self, e2e, ttft, tbt_avg, gpu_mem, out_len, in_len):
        self.e2e_latencies.append(e2e)
        if ttft is not None: self.ttfts.append(ttft)
        if tbt_avg is not None: self.tbts.append(tbt_avg)
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

        print(f"\n===== Performance Report =====")
        print(f"Total Samples: {len(self.e2e_latencies)}")
        print(f"E2E Latency (s): P50={stats['e2e_p50']:.4f}, P95={stats['e2e_p95']:.4f}")
        print(f"TTFT (s):        P50={stats['ttft_p50']:.4f}, P95={stats['ttft_p95']:.4f}, Avg={stats['ttft_avg']:.4f}")
        print(f"TBT (s):         P50={stats['tbt_p50']:.4f}, P95={stats['tbt_p95']:.4f}, Avg={stats['tbt_avg']:.4f}")
        print(f"Decode TPS:      P50={stats['tps_p50']:.2f}, Avg={stats['tps_avg']:.2f}")
        print(f"Peak VRAM (MB):  Max={stats['vram_max_mb']:.2f}")
        print(f"Avg Output Len:  {stats['avg_output_len']:.1f} tokens")
        print(f"Avg Input Len:   {stats['avg_input_len']:.1f} tokens")
        print("==============================")
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

# Timing Processor
class TimingLogitsProcessor(LogitsProcessor):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="benchmark_data.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model-path", type=str, default="checkpoints/vlm_checkpoints/llava-v1.5-7b", help="Path to model checkpoint")
    parser.add_argument("--report-file", type=str, default=None, help="Path to save the JSON report")
    args = parser.parse_args()

    # 1. Setup DToMe Configuration
    threshold_checkpoint_path = "checkpoints/threshold_checkpoints/ViT-L-14-336-tome-72out.pth"
    model_path = args.model_path 
    model_name = "llava_v1.5_tome_openai"

    learned_thresholds = find_thresholds(threshold_checkpoint_path)
    tome_kwargs = {
        'pretrained': "openai",
        'pretrained_origin_tag': "openai",
        'merge_mode': "batch_level",
        "repeat_merged_tokens": True,
        "r_total": 504,
        "specified_thresholds": learned_thresholds
    }
    print(f"Loaded DToMe thresholds: {len(learned_thresholds)}")

    # 2. Load Model
    print(f"Loading model: {model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        tome_kwargs=tome_kwargs,
        device_map="cuda"
    )

    # 3. Load Data
    with open(args.data_file, 'r') as f:
        file_content = json.load(f)
    if isinstance(file_content, dict) and 'data' in file_content:
        data_items = file_content['data']
        metadata = file_content.get('meta', {})
    else:
        data_items = file_content
        metadata = {}
    
    print(f"Loaded {len(data_items)} items from {args.data_file}")

    metrics = PerformanceMetrics(metadata)
    conv_mode = "llava_v1"

    # 4. Run Benchmark
    pbar = tqdm(data_items)
    for i, item in enumerate(pbar):
        torch.cuda.reset_peak_memory_stats()
        
        # Prepare Prompt & Images Same as benchmark_perf.py
        prompt_struct = item['prompt_struct']
        images = []
        curr_text = ""
        for p in prompt_struct:
            if p['type'] == 'image':
                img_path = p['value']
                # Try to handle relative path from perf_data root if needed, 
                # but standard benchmark scripts usually expect correct paths or relative to CWD.
                # Assuming inputs are correct absolute paths or relative to execution dir.
                if not os.path.exists(img_path) and not os.path.isabs(img_path):
                     # fallback: try prefixing with vlm_opt_linux if running from dymu
                     candidate = os.path.join("/home/user/vlm_opt_linux", img_path)
                     if os.path.exists(candidate):
                         img_path = candidate
                
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                    curr_text += DEFAULT_IMAGE_TOKEN + "\n" # specific for llava_v1
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            elif p['type'] == 'text':
                curr_text += p['value']
        
        # Remove last newline if added by image
        curr_text = curr_text.strip()

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], curr_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if images:
            image_tensor = process_images(images, image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        else:
            image_tensor = None
            
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_len = input_ids.shape[1]

        # Generation
        torch.cuda.synchronize()
        t_start = time.time()
        timing_processor = TimingLogitsProcessor(t_start)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                logits_processor=[timing_processor]
            )
            
        torch.cuda.synchronize()
        t_end = time.time()

        # Metrics
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
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2) 
        metrics.update(e2e, ttft, tbt_avg, peak_mem, output_len, input_len)
        pbar.set_postfix({'E2E': f"{e2e:.2f}s", 'TPS': f"{1/tbt_avg if tbt_avg>0 else 0:.1f}"})

    if args.report_file:
        metrics.save_json(args.report_file)
    else:
        metrics.report()

if __name__ == "__main__":
    main()
