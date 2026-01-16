import sys
import os

# Mock classes/functions to mimic correct behavior without needing full environment if strictly unit testing logic
# but better to use real modules if possible. 
# We will use real modules from LLaVA.

from llava.conversation import conv_templates, SeparatorStyle

def get_vlmeval_prompt():
    # Mimic vlmeval/vlm/llava/llava.py logic EXACTLY
    
    # 1. System Prompt
    system_prompt = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
    )
    
    # 2. Concat tilist logic
    # message = [{"type": "image", "value": "..."}, {"type": "text", "value": "Please describe this image."}]
    # In vlmeval, generated from dataset usually:
    # prompt += " <image>\n"  <-- Note the leading space in line 133 of llava.py
    
    text = ""
    # Image part
    text += " <image>\n"
    # Text part
    text += "Please describe this image."
    
    content = text 
    # generate_inner line 207:
    prompt = system_prompt + "USER: " + content + " ASSISTANT: "
    return prompt

def get_dymu_prompt():
    # Mimic dymu/run_benchmark_dymu.py logic EXACTLY
    
    conv_mode = "llava_v1"
    
    # 1. content construction
    # curr_text += DEFAULT_IMAGE_TOKEN + "\n"
    # curr_text += value
    DEFAULT_IMAGE_TOKEN = "<image>"
    curr_text = ""
    curr_text += DEFAULT_IMAGE_TOKEN + "\n"
    curr_text += "Please describe this image."
    
    # 2. whitespace strip
    curr_text = curr_text.strip()
    
    # 3. Template
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], curr_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    return prompt

if __name__ == "__main__":
    p_vlm = get_vlmeval_prompt()
    p_dymu = get_dymu_prompt()
    
    print(f"VLMEval Length: {len(p_vlm)}")
    print(f"DyMU    Length: {len(p_dymu)}")
    
    if p_vlm == p_dymu:
        print("MATCH!")
    else:
        print("MISMATCH!")
        print("\n--- VLMEval ---")
        print(repr(p_vlm))
        print("\n--- DyMU ---")
        print(repr(p_dymu))
        
        # Find index of first difference
        min_len = min(len(p_vlm), len(p_dymu))
        for i in range(min_len):
            if p_vlm[i] != p_dymu[i]:
                print(f"\nFirst difference at index {i}:")
                print(f"VLMEval char: {repr(p_vlm[i])}")
                print(f"DyMU    char: {repr(p_dymu[i])}")
                print(f"Context: ...{p_vlm[i-10:i+10]}...")
                break
