"""
Diagnostic script to check LLaVA model structure and compatibility
"""
import torch
from transformers import AutoConfig

model_path = "llava-hf/llava-1.5-7b-hf"

print("=" * 80)
print(f"Checking model: {model_path}")
print("=" * 80)

try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("\n✓ Successfully loaded config")
    print(f"\nModel type: {config.model_type}")
    print(f"\nConfig attributes:")
    for attr in sorted(dir(config)):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            value = getattr(config, attr, None)
            if value is not None and not isinstance(value, (dict, list)):
                print(f"  {attr}: {value}")
    
    print("\n" + "=" * 80)
    print("Checking for vision tower configuration:")
    print("=" * 80)
    
    # Check for various vision tower attributes
    vision_attrs = ['mm_vision_tower', 'vision_tower', 'vision_config', 'vision_model']
    for attr in vision_attrs:
        if hasattr(config, attr):
            print(f"✓ Found: {attr} = {getattr(config, attr)}")
        else:
            print(f"✗ Not found: {attr}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    
    if hasattr(config, 'mm_vision_tower'):
        print("✓ This model has the expected 'mm_vision_tower' attribute")
        print("  The model should work with the current FastV code")
    else:
        print("✗ This model does NOT have 'mm_vision_tower' attribute")
        print("  This is a Hugging Face format model, not the original LLaVA format")
        print("\nPossible solutions:")
        print("  1. Use the original LLaVA model format from liuhaotian/llava-v1.5-7b")
        print("  2. Convert the HF model to LLaVA format")
        print("  3. Modify FastV code to support HF format (recommended)")
        
except Exception as e:
    print(f"\n✗ Error loading config: {e}")
    import traceback
    traceback.print_exc()
