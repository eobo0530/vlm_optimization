import torch
from typing import Tuple, Callable, List, Optional, Union
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaModel, 
    _make_causal_mask, 
    _expand_mask
)
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast


class FastVLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        super().__init__(config)
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # build causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if past_key_values is not None:
            # past_key_values is a list of tuples (key, value)
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Llama 4.31.0 doesn't have _use_flash_attention_2 or _use_sdpa in the same way 4.36 does
        # It uses a simpler mask preparation
        if attention_mask is None:
             attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), dtype=torch.bool, device=inputs_embeds.device
            )
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Custom forward for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                # Use config values if available, otherwise default to hardcoded values
                K = getattr(self.config, "fast_v_k", 3)
                ratio = getattr(self.config, "fast_v_ratio", 0.5)
                sys_len = getattr(self.config, "fast_v_sys_length", 35)
                img_len = getattr(self.config, "fast_v_image_token_length", 576)

                if idx == K and seq_length > 1:
                    device = hidden_states.device
                    curr_len = self.last_attention.size(-1)
                    actual_sys_len = min(sys_len, curr_len)
                    actual_img_end = min(sys_len + img_len, curr_len)
                    
                    # image_attention_score indices: actual_sys_len to actual_img_end
                    image_attention_score = self.last_attention.mean(dim=1)[0][-1][actual_sys_len:actual_img_end]  
                    
                    if image_attention_score.size(0) > 0:
                        k_top = min(int(img_len * ratio), image_attention_score.size(0))
                        top_attention_rank_index = image_attention_score.topk(k_top).indices + actual_sys_len
                        keep_indexs = torch.cat((torch.arange(actual_sys_len, device=device), top_attention_rank_index, torch.arange(actual_img_end, seq_length, device=device)))
                    else:
                        keep_indexs = torch.arange(seq_length, device=device)
                    
                    keep_indexs = keep_indexs.sort().values
                    hidden_states = hidden_states[:,keep_indexs,:]
                    
                    # CRITICAL: Prune the KV cache of previous layers too
                    if use_cache and next_decoder_cache is not None:
                        new_next_decoder_cache = ()
                        for i in range(len(next_decoder_cache)):
                            k_cache, v_cache = next_decoder_cache[i]
                            # cache shape: (bsz, num_heads, seq_len, head_dim)
                            # CRITICAL: Use .contiguous() to ensure GPU efficiency
                            new_next_decoder_cache += ((k_cache[:, :, keep_indexs, :].contiguous(), v_cache[:, :, keep_indexs, :].contiguous()),)
                        next_decoder_cache = new_next_decoder_cache

                    # Update mask and position_ids
                    new_seq_length = keep_indexs.shape[0]
                    attention_mask = self._prepare_decoder_attention_mask(
                        None, (batch_size, new_seq_length), inputs_embeds, 0
                    )
                    position_ids = keep_indexs.unsqueeze(0)

                if idx == K - 1:
                    temp_layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=True,
                        use_cache=use_cache,
                    )
                    self.last_attention = temp_layer_outputs[1]

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

