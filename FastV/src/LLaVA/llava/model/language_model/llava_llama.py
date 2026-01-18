#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaMLP, LlamaRMSNorm, repeat_kv

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# --- KERNEL OPTIMIZATIONS ---

def scatter_mean(src, index, dim, dim_size):
    """Optimized scatter_mean without CPU-GPU synchronization, supporting 1D index for 3D src."""
    index = index.to(src.device)
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape, dtype=src.dtype)
    count = src.new_zeros(out_shape, dtype=src.dtype)
    
    if index.dim() < src.dim():
        # Correct expansion for [bsz, q_len, dim]
        index = index.view(1, -1, 1).expand_as(src)
        
    out.scatter_add_(dim, index, src)
    count.scatter_add_(dim, index, torch.ones_like(src))
    return out / count.clamp(min=1)

def remerge_mapping_hidden_states(hidden_states, mapping_indices, dim_size):
    return scatter_mean(hidden_states, mapping_indices, dim=1, dim_size=dim_size)

class LlavaConfig(LlamaConfig):
    model_type = "llava"

class CustomLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int = None):
        super().__init__(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        mapping_indices: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # Handle position_ids shape safely
        _pos = position_ids.view(-1)
        cos = cos.squeeze(1).squeeze(0)[_pos].view(bsz, 1, q_len, self.head_dim)
        sin = sin.squeeze(1).squeeze(0)[_pos].view(bsz, 1, q_len, self.head_dim)

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        is_agg_layer = (output_attentions and self.layer_idx == getattr(self.config, 'fast_v_agg_layer', 3) - 1)

        if not is_agg_layer:
            if attention_mask is not None:
                # SAFE MASK CONVERSION: 0.0 is keep, negative is mask. 
                # SDPA expects True for keep, False for mask.
                attention_mask = (attention_mask == 0)
                
                # FastV/Pruning alignment during generation
                if attention_mask.shape[-1] < key_states.shape[-2]:
                    num_keep = attention_mask.shape[-1]
                    key_states = key_states[:, :, -num_keep:, :]
                    value_states = value_states[:, :, -num_keep:, :]
            
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False
            )
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # DyMU Remerge Logic
        if mapping_indices is not None and not is_agg_layer and mapping_indices.shape[0] == q_len:
            dim_size = mapping_indices.max().item() + 1
            attn_output = remerge_mapping_hidden_states(attn_output, mapping_indices, dim_size)
            attn_output = attn_output.view(bsz, dim_size, self.hidden_size)
        else:
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int = None):
        super().__init__(config)
        self.self_attn = CustomLlamaAttention(config=config, layer_idx=layer_idx)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                output_attentions=False, use_cache=False, mapping_indices=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            mapping_indices=mapping_indices, **kwargs
        )
        # Align residual with merged hidden_states if merging occurred
        if mapping_indices is not None and hidden_states.shape[1] != residual.shape[1] and mapping_indices.shape[0] == residual.shape[1]:
            residual = remerge_mapping_hidden_states(residual, mapping_indices, hidden_states.shape[1])
            
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        if use_cache: outputs += (present_key_value,)
        return outputs

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig
    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        self.layers = nn.ModuleList([CustomLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, 
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, 
                return_dict=None, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        mapping_indices = kwargs.pop('mapping_indices', None)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states: all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                output_attentions=output_attentions, use_cache=use_cache,
                mapping_indices=mapping_indices, **kwargs
            )
            hidden_states = layer_outputs[0]
            if use_cache: next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions: all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns,
        )

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaLlamaModel(config)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, 
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, 
                output_hidden_states=None, images=None, return_dict=None, **kwargs):
        if inputs_embeds is None:
            res = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )
            if len(res) == 10:
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, \
                impos, pos_trans, pos_track, mapping_indices = res
                kwargs.update({'impos': impos, 'pos_trans': pos_trans, 'pos_track': pos_track, 'mapping_indices': mapping_indices})
            else:
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = res

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, **kwargs
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        # Clamp logits for stability against piece id out of range
        if logits.shape[-1] > 32000:
            logits = logits[..., :32000]
            
        logits = logits.float()

        return CausalLMOutputWithPast(
            logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None: _inputs['images'] = images
        # Propagate DyMU metadata
        for k in ['impos', 'pos_trans', 'pos_track', 'mapping_indices']:
            if k in kwargs: _inputs[k] = kwargs[k]
        return _inputs

try:
    AutoConfig.register("llava", LlavaConfig)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
except ValueError:
    pass