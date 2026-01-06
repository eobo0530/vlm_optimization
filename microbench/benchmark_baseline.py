import argparse, json, time, os
import torch
import numpy as np
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2-VL-7B-Instruct")
    # p.add_argument("--data", required=True)
    # p.add_argument("--output", required=True)
    p.add_argument("--data", default="runs/assets/mmbench20/workload_vision20.jsonl") 
    p.add_argument("--output", default=f"runs/baseline_{time.strftime('%Y-%m-%d')}/baseline_results_qwen.json")
    p.add_argument("--decode_tokens", type=int, default=128)
    p.add_argument("--attn", default="sdpa", choices=["sdpa","eager","flash_attention_2"])
    p.add_argument("--dtype", default="fp16", choices=["fp16","bf16","auto"])
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()

def now():
    return time.perf_counter()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    print(f"Loading model from {args.model_path} with {args.attn} attn...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.attn,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading workload from {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        workload = [json.loads(line) for line in f if line.strip()]

    results = []
    prefill_ms_list, tps_list, vram_list = [], [], []

    @torch.inference_mode()
    def run_one(item):
        # 멀티이미지 지원
        images = item.get("images", [])
        for p in images:
            if not os.path.exists(p):
                print(f"Skipping {item.get('id')} - {p} not found")
                return None

        content = [{"type":"image","image":p} for p in images] + [{"type":"text","text":item["prompt"]}]
        messages = [{"role":"user","content":content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(device)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # 1) Prefill
        t0 = now()
        out = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (now() - t0) * 1000.0

        # 2) Decode TPS (past_key_values 루프)
        past = out.past_key_values
        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)

        attn = inputs.get("attention_mask", None)
        if attn is None:
            attn = torch.ones((next_id.shape[0], inputs["input_ids"].shape[1]), device=device, dtype=torch.long)

        torch.cuda.synchronize()
        t1 = now()
        for _ in range(args.decode_tokens):
            attn = torch.cat([attn, torch.ones((attn.shape[0], 1), device=device, dtype=attn.dtype)], dim=1)
            out2 = model(input_ids=next_id, attention_mask=attn, past_key_values=past, use_cache=True)
            past = out2.past_key_values
            next_id = out2.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        decode_s = now() - t1
        tps = args.decode_tokens / max(decode_s, 1e-9)

        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        return prefill_ms, tps, peak_gb

    # Warmup
    print(f"Warming up for {args.warmup} items...")
    for i in range(min(args.warmup, len(workload))):
        _ = run_one(workload[i])

    print(f"Starting benchmark on {len(workload)} items...")
    from tqdm import tqdm
    for item in tqdm(workload):
        r = run_one(item)
        if r is None:
            continue
        prefill_ms, tps, peak_gb = r
        prefill_ms_list.append(prefill_ms)
        tps_list.append(tps)
        vram_list.append(peak_gb)
        results.append({
            "id": item.get("id",""),
            "prefill_ms": prefill_ms,
            "decode_tps": tps,
            "peak_vram_gb": peak_gb,
            "num_images": len(item.get("images", [])),
        })

    def pct(xs, q):
        xs = sorted(xs)
        k = int(round((len(xs)-1)*q))
        return xs[k]

    report = {
        "n": len(results),
        "prefill_ms_p50": pct(prefill_ms_list, 0.50) if prefill_ms_list else 0,
        "prefill_ms_p95": pct(prefill_ms_list, 0.95) if prefill_ms_list else 0,
        "decode_tps_p50": pct(tps_list, 0.50) if tps_list else 0,
        "decode_tps_p95": pct(tps_list, 0.95) if tps_list else 0,
        "peak_vram_gb_p95": pct(vram_list, 0.95) if vram_list else 0,
        "items": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({k:v for k,v in report.items() if k!="items"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
