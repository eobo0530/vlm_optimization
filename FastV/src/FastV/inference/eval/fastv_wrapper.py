"""
FastV wrapper for VLMEvalKit integration.
This module provides a custom LLaVA model class that injects FastV configuration.
"""
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '../../..'))  # Add src/ to path

import torch
from PIL import Image
import warnings
from abc import abstractproperty

# Import VLMEvalKit base class
try:
    from vlmeval.vlm.base import BaseModel
    from vlmeval.smp import *
    from vlmeval.dataset import DATASET_TYPE
except ImportError as e:
    raise ImportError(
        "Please ensure VLMEvalKit is installed and accessible. "
        f"Original error: {e}"
    )

# Import LLaVA dependencies
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
        "Please ensure LLaVA is installed. "
        f"Original error: {e}"
    )


class FastVLLaVA(BaseModel):
    """
    FastV-enabled LLaVA model for VLMEvalKit.
    
    This class extends the standard LLaVA model to support FastV token pruning.
    """
    
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path="liuhaotian/llava_v1.5_7b",
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
            
            logging.info("FastV configuration injected:")
            logging.info(f"  - use_fast_v: {self.use_fast_v}")
            logging.info(f"  - fast_v_inplace: {fast_v_inplace}")
            logging.info(f"  - fast_v_sys_length: {fast_v_sys_length}")
            logging.info(f"  - fast_v_image_token_length: {fast_v_image_token_length}")
            logging.info(f"  - fast_v_attention_rank: {fast_v_attention_rank}")
            logging.info(f"  - fast_v_agg_layer: {fast_v_agg_layer}")
        else:
            self.model.config.use_fast_v = False
            logging.info("FastV disabled")
        
        # Reset FastV state
        self.model.model.reset_fastv()
        
        # Set generation kwargs
        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def use_custom_prompt(self, dataset):
        """Check if we should use custom prompt for this dataset."""
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        """Build prompt for MCQ-style datasets."""
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
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output
