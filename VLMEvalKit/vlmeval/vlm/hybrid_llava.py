import torch
import torch.nn as nn
import os
from .llava.llava import LLaVA
from ..smp import *
from ..dataset import build_dataset

# Default threshold checkpoint path for DyMU
DEFAULT_THRESHOLD_PATH = "/home/aips/vlm/checkpoints/threshold_checkpoints/ViT-L-14-336-tome-72out.pth"

def find_thresholds(threshold_finding_checkpoint):
    """Load learned thresholds from a DyMU checkpoint."""
    import open_clip
    print(f"Loading DyMU thresholds from {threshold_finding_checkpoint}")
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

class HybridLLaVA(LLaVA):
    INTERLEAVE = False

    def __init__(self, model_path='liuhaotian/llava-v1.5-7b', **kwargs):
        # Load learned thresholds if checkpoint exists
        threshold_path = kwargs.pop('threshold_path', DEFAULT_THRESHOLD_PATH)
        
        if threshold_path and os.path.exists(threshold_path):
            learned_thresholds = find_thresholds(threshold_path)
            tome_kwargs = {
                'pretrained': 'openai',
                'pretrained_origin_tag': 'openai',
                'merge_mode': 'batch_level',
                'repeat_merged_tokens': False,  # Changed from True to False for logical mapping
                'r_total': 504,
                'specified_thresholds': learned_thresholds
            }
            print(f"Hybrid: Loaded {len(learned_thresholds)} DyMU thresholds from checkpoint. Logical mapping enabled.")
        else:
            # Fallback to constant schedule if no checkpoint found
            print(f"Warning: Threshold checkpoint not found at {threshold_path}, using constant schedule")
            tome_kwargs = kwargs.pop('tome_kwargs', {
                'pretrained': 'openai',
                'pretrained_origin_tag': 'openai',
                'r_total': 504,
                'merge_mode': 'batch_level',
                'r_schedule': 'constant',
                'repeat_merged_tokens': False  # Changed from True to False for logical mapping
            })
        
        super().__init__(model_path=model_path, **kwargs)
        
        self.vtu_enabled = tome_kwargs.get('repeat_merged_tokens', True)
        
        # num_beams will be set dynamically per-dataset in generate_inner:
        # - COCO/captioning: num_beams=5 (quality matters, short prompts so still fast)
        # - MMBench/MCQ: num_beams=1 (long prompts make beam search very slow)
        
        # FastV configuration
        env_rank = os.environ.get('FASTV_K', None)
        # Handle float for ratio-based pruning (e.g., 0.5 means 50% pruned)
        default_rank = float(env_rank) if env_rank is not None else 0.5

        self.model.config.use_fast_v = kwargs.get('use_fast_v', True)
        self.model.config.fast_v_inplace = kwargs.get('fast_v_inplace', True)
        self.model.config.fast_v_agg_layer = kwargs.get('fast_v_agg_layer', 3)
        self.model.config.fast_v_attention_rank = kwargs.get('fast_v_rank', default_rank)  # Correct attribute name
        self.model.config.fast_v_rank = kwargs.get('fast_v_rank', default_rank)  # Keep for backwards compat
        
        # Align FastV with DyMU's merged token count
        # If repeat_merged_tokens is True (VTU ON), token count is 576
        # If False (VTU OFF), token count is 72 (576 - 504)
        if tome_kwargs.get('repeat_merged_tokens', True):
            self.model.config.fast_v_image_token_length = 576
        else:
            self.model.config.fast_v_image_token_length = 72
        
        # System prompt length (before image tokens). Default 35 for typical LLaVA prompts.
        self.model.config.fast_v_sys_length = kwargs.get('fast_v_sys_length', 35)
        
        # Synchronize config to model instance (LlamaModel copies config in __init__)
        if hasattr(self.model, 'model'):
            llama_model = self.model.model
            if hasattr(llama_model, 'reset_fastv'):
                llama_model.reset_fastv()
            else:
                # Manual sync if reset_fastv doesn't exist
                for attr in ['use_fast_v', 'fast_v_inplace', 'fast_v_agg_layer', 
                            'fast_v_attention_rank', 'fast_v_sys_length', 'fast_v_image_token_length']:
                    if hasattr(self.model.config, attr):
                        setattr(llama_model, attr, getattr(self.model.config, attr))
        
        # NOTE: Do NOT set config.output_attentions = True globally!
        # The Conditional SDPA logic in LlamaModel.forward handles output_attentions
        # per-layer (only layer AGG_LAYER-1 needs attention weights for FastV).
        # Setting it globally forces ALL layers to use slow manual attention.
        
        if self.model.config.use_fast_v:
            print(f"Hybrid: FastV enabled (rank={self.model.config.fast_v_attention_rank}, image_len={self.model.config.fast_v_image_token_length}, sys_len={self.model.config.fast_v_sys_length})")
        else:
            print("Hybrid: FastV disabled")
            
        # [Optimization Fix] Force replace vision tower with ToMe version (DyMU)
        # Because LLaVA parent class and FastV builder do not handle tome_kwargs automatically.
        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTowerToMe
        vision_tower = self.model.get_vision_tower()
        
        if not isinstance(vision_tower, CLIPVisionTowerToMe):
            print(f"Hybrid: Vanilla Vision Tower detected. Swapping to CLIPVisionTowerToMe via Hot-swap...")
            
            vision_tower_name = vision_tower.vision_tower_name
            if "openai/clip-vit-large-patch14-336" in vision_tower_name:
                vision_tower_name = "ViT-L-14-336-tome-72out"

            # Create new ToMe tower
            # CLIPVisionTowerToMe.__init__ does not accept kwargs, so we pass kwargs to load_model
            new_tower = CLIPVisionTowerToMe(vision_tower_name, args=self.model.config, delay_load=True)
            new_tower.load_model(device="cuda" if torch.cuda.is_available() else "cpu", dtype=self.model.dtype, **tome_kwargs)
            
            # Replace in model
            self.model.model.vision_tower = new_tower
            
            # Update image_processor
            self.image_processor = new_tower.image_processor
            self.model.get_vision_tower().image_processor = new_tower.image_processor
            print(f"Hybrid: DyMU Vision Tower injected successfully.")
            

    def generate_inner(self, message, dataset=None):
        from ..dataset import DATASET_TYPE
        
        # Dynamic num_beams based on dataset type
        # MCQ datasets have long prompts -> beam search is very slow
        # Caption datasets have short prompts -> beam search is acceptable
        if dataset and DATASET_TYPE(dataset) == 'MCQ':
            self.kwargs['num_beams'] = 1  # Greedy for MCQ (fast)
        else:
            self.kwargs['num_beams'] = 5  # Beam search for captioning (quality)
        
        # Ensure output_attentions is passed through generate for the aggregation layer
        # This is CRITICAL for FastV to work
        self.kwargs['output_attentions'] = True
        
        # Reset FastV state machine for new generation
        # Try both common naming conventions
        if hasattr(self.model, 'reset_fastv'):
            self.model.reset_fastv()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'reset_fastv'):
            self.model.model.reset_fastv()
        elif hasattr(self.model, 'reset_fast_v'):
            self.model.reset_fast_v()
        
        print(f"Hybrid: Generating with FASTV_K={self.model.config.fast_v_attention_rank}, Logical Mapping=ON, output_attentions={self.kwargs.get('output_attentions')}")
            
        try:
            output_ids = super().generate_inner(message, dataset=dataset)
            return output_ids
        except IndexError as e:
            # We can't easily get output_ids here if it's trapped in the super call
            # But let's look at the tokenizer again
            print(f"IndexError caught! Tokenizer vocab size: {len(self.tokenizer)}")
            raise e

