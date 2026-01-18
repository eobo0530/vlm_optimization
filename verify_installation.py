
import importlib.util
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    
    try:
        if module_name == "cv2":
            import cv2
            module = cv2
        else:
            module = importlib.import_module(module_name)
        
        if module_name == "transformers":
             print(f"   path: {module.__file__}")

        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {display_name} installed (Version: {version})")
        return True
    except ImportError as e:
        print(f"[FAIL] {display_name} NOT installed. Error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] {display_name} error during import. Error: {e}")
        return False

# OpenCLIP Debug
try:
    import open_clip
    print(f"DEBUG: open_clip file: {open_clip.__file__}")
    try:
        import open_clip.tome
        print("DEBUG: open_clip.tome imported successfully")
    except ImportError as e:
        print(f"DEBUG: open_clip.tome import FAILED: {e}")
except ImportError as e:
    print(f"DEBUG: open_clip import FAILED: {e}")

packages_to_check = [
    ("transformers", "Transformers (FastV Patched)"),
    ("llava", "LLaVA"),
    ("dymu", "DyMU"),
    ("vlmeval", "VLMEvalKit"),
    ("httpx", "httpx"),
    ("pydantic", "pydantic"),
    ("cv2", "opencv-python"), 
    ("sentence_transformers", "sentence-transformers"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn")
]

print("Verifying installation...")
all_success = True
for module, name in packages_to_check:
    if not check_import(module, name):
        all_success = False

if all_success:
    print("\nAll key components are successfully installed!")
    sys.exit(0)
else:
    print("\nSome components failed to load.")
    sys.exit(1)
