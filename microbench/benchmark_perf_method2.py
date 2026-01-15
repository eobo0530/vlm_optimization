
import os
import json
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import LogitsProcessor
from types import MethodType

# Try importing LLaVA modules
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
except ImportError as e:
    print(f"Error: LLaVA not found. Please ensure it is installed. Details: {e}")
    exit(1)

from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union

###############################################################################
# MONKEY PATCHING LOGIC (Method 2: Static KV-Cache Pruning)
# Robust Implementation: Prunes KV Cache within Attention
###############################################################################

def fastv_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()

    # FastV parameters from parent model/config
    use_fast_v = getattr(self.config, "use_fast_v", False)
    K = getattr(self.config, "fast_v_agg_layer", 3)
    ratio = 1.0 - (getattr(self.config, "fast_v_attention_rank", 288) / 576.0)
    sys_len = getattr(self.config, "fast_v_sys_length", 35)
    img_len = getattr(self.config, "fast_v_image_token_length", 576)
    
    # We are in prefill if q_len > 1
    is_prefill = q_len > 1

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # RoPE
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # FASTV PRUNING LOGIC
    if use_fast_v and is_prefill and self.layer_idx == K:
        # We need the attention weights from layer K-1
        last_attn = getattr(self, "_last_attention_score", None)
        if last_attn is not None:
            current_device = query_states.device
            # last_attn shape: [bs, heads, q_len, kv_len]
            # Use the attention of the last token towards all previous tokens
            avg_attn = last_attn.mean(dim=1)[0][-1] 
            
            img_start = sys_len
            img_end = sys_len + img_len
            # Clamp to actual attention size
            img_start = min(img_start, avg_attn.size(0))
            img_end = min(img_end, avg_attn.size(0))
            
            img_scores = avg_attn[img_start:img_end]
            if img_scores.size(0) > 0:
                num_keep = int(img_scores.size(0) * (1 - ratio))
                top_indices = img_scores.topk(num_keep).indices + img_start
                keep_indices = torch.cat((
                    torch.arange(img_start, device=current_device),
                    top_indices,
                    torch.arange(img_end, avg_attn.size(0), device=current_device)
                )).sort().values
                
                # Prune Key and Value states
                # key_states/value_states shape: [bs, heads, seq, dim]
                key_states = key_states[:, :, keep_indices, :]
                value_states = value_states[:, :, keep_indices, :]
                
                # Prune attention mask: [bs, 1, q_len, kv_len]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, keep_indices]
                
                kv_seq_len = key_states.shape[-2]

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Standard Attention
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(self.head_dim)
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Store attention for next layer if needed
    if use_fast_v and is_prefill and self.layer_idx == K - 1:
        # Store in the next layer's attention module
        next_layer_attn = self._next_attention_module
        if next_layer_attn is not None:
            next_layer_attn._last_attention_score = attn_weights.detach()

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights if output_attentions else None, past_key_value

# Helper for RoPE (copied from transformers to ensure availability in monkey patch)
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

###############################################################################
# Performance Reporting Logic
###############################################################################

# Metric Collector (Same as before)
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
            "e2e_p50": p50(self.e2e_latencies), "e2e_p95": p95(self.e2e_latencies),
            "ttft_p50": p50(self.ttfts), "ttft_p95": p95(self.ttfts), "ttft_avg": avg(self.ttfts),
            "tps_avg": avg([1.0/t if t > 0 else 0 for t in self.tbts]),
            "vram_max_mb": max(self.gpu_mem_peaks) if self.gpu_mem_peaks else 0,
            "avg_output_len": avg(self.output_lens)
        }
        print(f"\n===== Performance Report (Method 2: Integrated KV Pruning) =====")
        print(f"Total Samples: {len(self.e2e_latencies)}")
        print(f"TTFT (s): Avg={stats['ttft_avg']:.4f}")
        print(f"Decode TPS: Avg={stats['tps_avg']:.2f}")
        print(f"Peak VRAM (MB): Max={stats['vram_max_mb']:.2f}")
        return stats

    def save_json(self, filename):
        stats = self.report()
        with open(filename, 'w') as f:
            json.dump({"summary": stats}, f, indent=4)

# Timing Processor
class TimingLogitsProcessor(LogitsProcessor):
    def __init__(self, start_time):
        self.start_time = start_time
        self.first_token_time = None
        self.token_times = []
    def __call__(self, input_ids, scores):
        now = time.time()
        if self.first_token_time is None: self.first_token_time = now
        self.token_times.append(now)
        return scores

def main(args):
    print(f"Loading model: {args.model_path}")
    model_path = args.model_path
    if model_path.endswith("_fastv"):
        model_path = "liuhaotian/llava-v1.5-7b"

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, model_base=None, model_name=model_name, device_map="cuda"
    )

    # Apply Monkey Patch to Attention Modules
    K = args.r
    print(f"🚀 Enabling FASTV METHOD 2 (Integrated KV Cache Pruning): K={args.k}, R={K}")
    
    # We need to link layer K-1 and layer K
    layers = model.model.layers
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        attn.layer_idx = i
        attn.config = model.config
        # Inject fastv params into config if not there
        model.config.use_fast_v = True
        model.config.fast_v_agg_layer = K
        model.config.fast_v_attention_rank = args.k
        model.config.fast_v_sys_length = 35
        model.config.fast_v_image_token_length = 576
        
        # Monkey patch
        attn.forward = MethodType(fastv_attention_forward, attn)
        
        # Linkages
        attn._next_attention_module = layers[i+1].self_attn if i+1 < len(layers) else None
        attn._last_attention_score = None

    print("✅ LlamaAttention.forward has been monkey-patched.")

    # Load Data
    with open(args.data_file, 'r') as f:
        data_items = json.load(f)['data']
    
    metrics = PerformanceMetrics()
    
    # Loop
    pbar = tqdm(data_items[:args.num_samples])
    for i, item in enumerate(pbar):
        torch.cuda.reset_peak_memory_stats()
        
        # Prepare Inputs
        prompt_struct = item['prompt_struct']
        images = []
        curr_text = ""
        for p in prompt_struct:
            if p['type'] == 'image':
                img_path = p['value']
                images.append(Image.open(img_path).convert("RGB"))
                curr_text += "<image>\n"
            elif p['type'] == 'text':
                curr_text += p['value']
        
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], curr_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16) if images else None
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_len = input_ids.shape[1]
        
        # Reset FastV temporary state in attention modules
        for layer in model.model.layers:
            layer.self_attn._last_attention_score = None

        # Generation
        torch.cuda.synchronize()
        t_start = time.time()
        timing_processor = TimingLogitsProcessor(t_start)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=image_tensor, do_sample=False, temperature=0,
                max_new_tokens=args.max_new_tokens, logits_processor=[timing_processor], use_cache=True
            )
            
        torch.cuda.synchronize()
        t_end = time.time()

        e2e = t_end - t_start
        output_len = output_ids.shape[1] - input_len
        ttft = timing_processor.first_token_time - t_start if timing_processor.first_token_time else 0
        tbt_avg = (timing_processor.token_times[-1] - timing_processor.first_token_time) / (output_len - 1) if output_len > 1 else 0
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        
        metrics.update(e2e, ttft, tbt_avg, peak_mem, output_len, input_len)
        pbar.set_postfix({'TPS': f"{1/tbt_avg if tbt_avg>0 else 0:.1f}"})
        
    if args.report_file:
        metrics.save_json(args.report_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--report-file", type=str, default=None)
    parser.add_argument("--k", type=int, default=288)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=500)
    args = parser.parse_args()
    main(args)
