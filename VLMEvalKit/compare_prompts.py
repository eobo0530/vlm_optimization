from llava.conversation import conv_templates

# Simulate VLMEval construction
system_prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
)
content = "<image>\nPlease describe this image."
# vlmeval/vlm/llava/llava.py line 207 (approx)
vlmeval_prompt = system_prompt + "USER: " + content + " ASSISTANT: "

# Simulate standard LLaVA / DyMU construction
conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], content)
conv.append_message(conv.roles[1], None)
dymu_prompt = conv.get_prompt()

print("=== VLMEval Prompt ===")
print(repr(vlmeval_prompt))
print("\n=== DyMU / Standard Prompt ===")
print(repr(dymu_prompt))

print("\n=== Match? ===")
print(vlmeval_prompt == dymu_prompt)

if vlmeval_prompt != dymu_prompt:
    print("\nDifferences:")
    import difflib
    for diff in difflib.unified_diff(vlmeval_prompt.splitlines(), dymu_prompt.splitlines()):
        print(diff)
