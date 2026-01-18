import torch
import torch.nn as nn
from .llava.llava import LLaVA
from ..smp import *
from ..dataset import build_dataset

class HybridLLaVA(LLaVA):
    INTERLEAVE = False

    def __init__(self, model_path='liuhaotian/llava-v1.5-7b', **kwargs):
        # Tome kwargs for DyMU
        tome_kwargs = kwargs.pop('tome_kwargs', {
            'r_total': 504,
            'merge_mode': 'batch_level',
            'r_schedule': 'constant'
        })
        
        super().__init__(model_path=model_path, **kwargs)
        
        # FastV configuration
        self.model.config.use_fast_v = kwargs.get('use_fast_v', True)
        self.model.config.fast_v_inplace = kwargs.get('fast_v_inplace', True)
        self.model.config.fast_v_agg_layer = kwargs.get('fast_v_agg_layer', 3)
        self.model.config.fast_v_rank = kwargs.get('fast_v_rank', 288)
        
        # Align FastV with DyMU's merged token count
        # 576 -> 72 (if r=504)
        # We use dymu_n_un if it was set in config, otherwise default to 72
        self.model.config.fast_v_image_token_length = getattr(self.model.config, 'dymu_n_un', 72)
        
        # Force output_attentions for FastV aggregation
        self.model.config.output_attentions = True
        
        if self.model.config.use_fast_v:
            print(f"Hybrid: FastV enabled (rank={self.model.config.fast_v_rank}, image_len={self.model.config.fast_v_image_token_length})")
        else:
            print("Hybrid: FastV disabled")
            

    def generate_inner(self, message, dataset=None):
        # Ensure output_attentions is passed through generate for the aggregation layer
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        
        # Reset FastV state machine for new generation
        if hasattr(self.model, 'reset_fast_v'):
            self.model.reset_fast_v()
            
        try:
            output_ids = super().generate_inner(message, dataset=dataset)
            return output_ids
        except IndexError as e:
            # We can't easily get output_ids here if it's trapped in the super call
            # But let's look at the tokenizer again
            print(f"IndexError caught! Tokenizer vocab size: {len(self.tokenizer)}")
            raise e
