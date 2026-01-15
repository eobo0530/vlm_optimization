"""
FastV wrapper for VLMEvalKit integration.
"""
# STEP 3: Now continue with normal imports
import torch
from PIL import Image
import warnings
import os
import os.path as osp
from abc import abstractproperty

# Import VLMEvalKit base class
from vlmeval.smp import *

# Import LLaVA dependencies (will now use FastV's LLaVA)
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError as e:
    raise ImportError(
        f"Please ensure LLaVA is installed and accessible. Original error: {e}"
    )


class FastVLLaVA:
    """
    FastV-enabled LLaVA model for VLMEvalKit.
    
    This class extends the standard LLaVA model to support FastV token pruning.
    """
    
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path="liuhaotian/llava-v1.5-7b",
        use_fast_v=True,
        fast_v_inplace=False,
        fast_v_sys_length=35,
        fast_v_image_token_length=576,
        fast_v_attention_rank=100,
        fast_v_agg_layer=3,
        **kwargs
    ):
        """
        Initialize FastV-enabled LLaVA model.
        
        Args:
            model_path: Path to the LLaVA model
            use_fast_v: Whether to enable FastV
            fast_v_inplace: Whether to use inplace FastV (affects latency measurement)
            fast_v_sys_length: Length of system prompt tokens
            fast_v_image_token_length: Number of image tokens (typically 576 for 24x24 patches)
            fast_v_attention_rank: Number of top attention tokens to keep
            fast_v_agg_layer: Layer index at which to aggregate attention scores
            **kwargs: Additional generation kwargs
        """
        assert osp.exists(model_path) or splitlen(model_path) == 2
        
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"
        
        # Handle special model names
        if model_path == "Lin-Chen/ShareGPT4V-7B":
            model_name = "llava-v1.5-7b"
        elif model_path == "Lin-Chen/ShareGPT4V-13B":
            model_name = "llava-v1.5-13b"
        else:
            model_name = get_model_name_from_path(model_path)
        
        # Load pretrained model
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device_map="cpu",
                )
            )
        except Exception as err:
            if "ShareGPT4V" in model_path:
                import llava
                logging.critical(
                    "Please manually remove the encoder type check in "
                    f"{llava.__path__[0]}/model/multimodal_encoder/builder.py "
                    "Line 8 to use the ShareGPT4V model. "
                )
            else:
                logging.critical("Unknown error when loading LLaVA model.")
            raise err
        
        self.model = self.model.cuda()
        self.conv_mode = "llava_v1"
        
        # Configure FastV
        self.use_fast_v = use_fast_v
        if self.use_fast_v:
            self.model.config.use_fast_v = True
            self.model.config.fast_v_inplace = fast_v_inplace
            self.model.config.fast_v_sys_length = fast_v_sys_length
            self.model.config.fast_v_image_token_length = fast_v_image_token_length
            self.model.config.fast_v_attention_rank = fast_v_attention_rank
            self.model.config.fast_v_agg_layer = fast_v_agg_layer
            self.model.config.output_attentions = True  # CRITICAL for FastV
            
            logger = get_logger('FastVLLaVA')
            logger.info("FastV configuration injected:")
            logger.info(f"  - use_fast_v: {self.use_fast_v}")
            logger.info(f"  - fast_v_inplace: {fast_v_inplace}")
            logger.info(f"  - fast_v_sys_length: {fast_v_sys_length}")
            logger.info(f"  - fast_v_image_token_length: {fast_v_image_token_length}")
            logger.info(f"  - fast_v_attention_rank: {fast_v_attention_rank}")
            logger.info(f"  - fast_v_agg_layer: {fast_v_agg_layer}")
        else:
            self.model.config.use_fast_v = False
            logger = get_logger('FastVLLaVA')
            logger.info("FastV disabled")
        
        # Reset FastV state (only if the method exists)
        if hasattr(self.model.model, 'reset_fastv'):
            self.model.model.reset_fastv()
        else:
            logger = get_logger('FastVLLaVA')
            logger.warning("reset_fastv() method not found - using standard LLaVA model")
        
        # Set generation kwargs
        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        # CRITICAL FIX: Update default dict with user kwargs so user values take precedence
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )
        
        # Initialize dump_image_func for VLMEvalKit compatibility
        self.dump_image_func = None
        
        # Verify FastV transformers is being used
        logger = get_logger('FastVLLaVA')
        import transformers
        logger.info(f"✅ Using transformers from: {transformers.__file__}")
        logger.info(f"✅ Model type: {type(self.model.model).__module__}.{type(self.model.model).__name__}")

    def set_dump_image(self, dump_image_func):
        """Set the dump_image function (required by VLMEvalKit)."""
        self.dump_image_func = dump_image_func
    
    def use_custom_prompt(self, dataset):
        """Check if we should use custom prompt for this dataset."""
        from vlmeval.dataset import DATASET_TYPE
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        """Build prompt for MCQ-style datasets."""
        import string
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message
    
    def dump_image(self, line, dataset):
        """Dump image from line (required by build_prompt)."""
        return self.dump_image_func(line)

    def concat_tilist(self, message):
        """Concatenate text and image list."""
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def generate_inner(self, message, dataset=None):
        """
        Core generation method called by VLMEvalKit.
        
        Args:
            message: List of dicts with 'type' and 'value' keys
            dataset: Dataset name (optional)
            
        Returns:
            Generated text output
        """
        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        
        if images:
            image_tensor = process_images(images, self.image_processor, args).to(
                "cuda", dtype=torch.float16
            )
        else:
            image_tensor = None

        if dataset == 'COCO_VAL':
            # Debugging found that COCO dataset already provides a specific prompt:
            # "Please describe this image in general. Directly provide the description, do not include prefix like..."
            # So we should validly use 'content' and NOT append "Please describe this image concisely."
            # Also, we skip system prompt for COCO as per LLaVA convention for captioning.
            prompt = "USER: " + content + " ASSISTANT:"
        else:
            prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        
        # Generate kwargs - add output_attentions for FastV
        gen_kwargs = self.kwargs.copy()
        if self.use_fast_v:
            gen_kwargs['output_attentions'] = True  # Required for FastV pruning
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **gen_kwargs,
            )

        # Slice output_ids to get only the generated tokens
        input_token_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output
    
    def generate(self, message, dataset=None):
        """
        Main generation interface called by VLMEvalKit.
        """
        return self.generate_inner(message, dataset)
