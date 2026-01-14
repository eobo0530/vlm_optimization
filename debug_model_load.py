
import os
import sys
import torch
import argparse
# Add LLaVA to path
sys.path.append(os.path.join(os.getcwd(), "LLaVA"))

try:
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
except ImportError:
    print("Failed to import CLIPVisionTower")
    sys.exit(1)

# Mock args
class MockArgs:
    def __init__(self, tower_name, tome_kwargs=None):
        self.mm_vision_tower = tower_name
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = 'patch'
        if tome_kwargs:
            self.tome_kwargs = tome_kwargs

model_name_dymu = "ViT-L-14-336-tome-72out"
tome_kwargs = {"dummy": True}

print(f"\n--- Testing Dymu Loading via CLIPVisionTower (with tome_kwargs) ---")
args = MockArgs(model_name_dymu, tome_kwargs=tome_kwargs)
try:
    tower = CLIPVisionTower(model_name_dymu, args=args)
    # Trigger load_model
    tower.load_model()
    print("Success! Tower loaded.")
    print(f"Loaded tower name: {tower.vision_tower_name}")
except Exception as e:
    print(f"Failed to load via CLIPVisionTower: {e}")

print(f"\n--- Testing Baseline Loading via CLIPVisionTower (no tome_kwargs) ---")
args_base = MockArgs(model_name_dymu, tome_kwargs=None)
try:
    tower = CLIPVisionTower(model_name_dymu, args=args_base)
    tower.load_model()
    print("Success! Tower loaded.")
    print(f"Loaded tower name: {tower.vision_tower_name}")
except Exception as e:
    print(f"Failed to load via CLIPVisionTower: {e}")
