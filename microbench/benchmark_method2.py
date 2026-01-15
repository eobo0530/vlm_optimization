import argparse
import time
import torch
import os
import sys
import json
from PIL import Image
import numpy as np
from types import MethodType

# Add FastV transformers and LLaVA to path
FASTV_TRANSFORMERS_PATH = '/home/aips/FastV/src/transformers/src'
LLAVA_PATH = '/home/aips/FastV/src/LLaVA'

for path in [FASTV_TRANSFORMERS_PATH, LLAVA_PATH]:
    if os.path.exists(path):
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added {path} to sys.path")

from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    print("Error: LLaVA library not found.")
    sys.exit(1)

###############################################################################
# MONKEY PATCHING LOGIC (Method 2: Static KV-Cache Pruning)
###############################################################################

def fastv_static_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    # Retrieve config values
    use_fast_v = getattr(self.config, "use_fast_v", False)
    K = getattr(self.config, "fast_v_agg_layer", 3)
    ratio = 1.0 - (getattr(self.config, "fast_v_attention_rank", 288) / 576.0) # Convert rank to ratio
    sys_len = getattr(self.config, "fast_v_sys_length", 35)
    img_len = getattr(self.config, "fast_v_image_token_length", 576)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    
    # Simple check for prefill (seq_length > 1)
    is_prefill = seq_length > 1

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # 1. Prepare mask
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
    
    seq_length_with_past = seq_length + past_key_values_length
    
    if position_ids is None:
        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
    
    causal_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    last_attention = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None
        
        # Method 2: Static Pruning Logic
        if use_fast_v and is_prefill:
            # If we are at the aggregation layer, perform pruning once
            if idx == K:
                current_device = hidden_states.device
                # Use attention scores from the previous layer
                # last_attention shape: [batch, heads, seq, seq]
                # We want the attention of the LAST token [ -1 ] towards all others
                avg_attn = last_attention.mean(dim=1)[0][-1] 
                
                img_start = sys_len
                img_end = sys_len + img_len
                
                img_scores = avg_attn[img_start:img_end]
                num_keep = int(img_len * (1 - ratio))
                
                top_indices = img_scores.topk(num_keep).indices + img_start
                # Keep sys prompt + top image tokens + tail tokens
                keep_indices = torch.cat((
                    torch.arange(img_start, device=current_device),
                    top_indices,
                    torch.arange(img_end, seq_length_with_past, device=current_device)
                )).sort().values
                
                # Physical Pruning
                hidden_states = hidden_states[:, keep_indices, :]
                position_ids = keep_indices.unsqueeze(0)
                new_len = keep_indices.size(0)
                
                # Re-prepare mask for the new shape
                causal_mask = self._prepare_decoder_attention_mask(
                    None, (batch_size, new_len), hidden_states, 0
                )

        # To get attention for the next layer's pruning decision
        layer_output_attentions = True if (use_fast_v and is_prefill and idx == K-1) else output_attentions

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=layer_output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]
        if layer_output_attentions:
            last_attention = layer_outputs[1]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if layer_output_attentions else 1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

###############################################################################
# BENCHMARK RUNNER
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Method 2 (Static KV-Cache Pruning) Micro-benchmark")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b", help="Model path")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the benchmark image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Benchmark prompt")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--use-fastv", action="store_true", help="Enable FastV Static Pruning")
    parser.add_argument("--k", type=int, default=288, help="FastV Tokens to keep (rank)")
    parser.add_argument("--r", type=int, default=3, help="FastV Aggregation Layer")
    return parser.parse_args()

def run_benchmark():
    args = parse_args()
    device = "cuda"
    model_name = get_model_name_from_path(args.model_path)
    
    print(f"Loading Model: {args.model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto"
    )

    if args.use_fastv:
        print(f"🚀 Enabling FASTV METHOD 2 (Static KV Pruning): K={args.k}, R={args.r}")
        model.config.use_fast_v = True
        model.config.fast_v_inplace = True # We use physical pruning
        model.config.fast_v_sys_length = 35
        model.config.fast_v_image_token_length = 576
        model.config.fast_v_attention_rank = args.k
        model.config.fast_v_agg_layer = args.r
        
        # MONKEY PATCH!
        model.model.forward = MethodType(fastv_static_forward, model.model)
        print("✅ LlamaModel.forward has been monkey-patched with Static Pruning logic.")

    # Prepare Input
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
    
    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{args.prompt} ASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    # Warmup
    print("Warming up...")
    for _ in range(2):
        with torch.inference_mode():
            model.generate(input_ids, images=image_tensor, max_new_tokens=5, do_sample=False, use_cache=True)

    print(f"Benchmarking (Generated Tokens: {args.max_new_tokens})...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.inference_mode():
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # 1. Prefill
        output = model.generate(
            input_ids, 
            images=image_tensor, 
            max_new_tokens=1, 
            do_sample=False, 
            use_cache=True,
            return_dict_in_generate=True
        )
        torch.cuda.synchronize()
        first_token_time = time.perf_counter()
        ttft = (first_token_time - start_time) * 1000
        
        # 2. Decode
        next_ids = output.sequences
        pk_values = output.past_key_values
        
        decode_start = time.perf_counter()
        final_output = model.generate(
            next_ids,
            images=image_tensor,
            past_key_values=pk_values,
            max_new_tokens=args.max_new_tokens - 1,
            do_sample=False,
            use_cache=True
        )
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
    total_decode_time = end_time - decode_start
    tps = (args.max_new_tokens - 1) / total_decode_time
    tbt = (total_decode_time * 1000) / (args.max_new_tokens - 1)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    print("\n" + "="*40)
    print(f" RESULTS: {'FASTV-STATIC' if args.use_fastv else 'BASELINE'} ")
    print("="*40)
    print(f"TTFT (Prefill)   : {ttft:.2f} ms")
    print(f"TBT (Decode)     : {tbt:.2f} ms/tok")
    print(f"TPS (Throughput) : {tps:.2f} tokens/s")
    print(f"Peak VRAM        : {peak_vram:.2f} MB")
    print(f"E2E Latency      : {end_time - start_time:.3f} s")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
