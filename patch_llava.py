import os

path = '/home/user/vlm_opt_linux/LLaVA/llava/model/language_model/llava_llama.py'
with open(path, 'r') as f:
    content = f.read()

target_sig = """        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:"""

replace_sig = """        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:"""

target_super = """            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )"""

replace_super = """            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )"""

if target_sig in content:
    new_content = content.replace(target_sig, replace_sig)
    if target_super in new_content:
        new_content = new_content.replace(target_super, replace_super)
        with open(path, 'w') as f:
            f.write(new_content)
        print("Patched successfully")
    else:
        print("Target super call not found")
else:
    print("Target signature not found")
