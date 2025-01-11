# check_specific_model.py
import os
from pathlib import Path

def check_specific_model():
    # Common base directories to check
    base_dirs = [
        Path(r"C:\Users\pawan\.cache\huggingface"),
        Path(r"C:\Users\pawan\.cache\huggingface\hub"),
        Path(r"C:\Users\pawan\.cache\huggingface\transformers"),
        Path(r"C:\Users\pawan\AppData\Local\huggingface"),
        Path(r"C:\Users\pawan\AppData\Roaming\huggingface")
    ]
    
    model_names = [
        "Llama-3.3-70B-Instruct",
        "Llama-2-7b-chat-hf",
        "llama"  # Generic search
    ]
    
    print("\nSearching for LLaMA models...")
    print("-" * 50)
    
    for base_dir in base_dirs:
        if base_dir.exists():
            print(f"\nChecking directory: {base_dir}")
            try:
                # Walk through all subdirectories
                for root, dirs, files in os.walk(base_dir):
                    root_path = Path(root)
                    for model_name in model_names:
                        if model_name.lower() in root.lower():
                            print(f"\nFound potential model files:")
                            print(f"Path: {root_path}")
                            print("\nContents:")
                            for item in os.listdir(root_path):
                                print(f"- {item}")
            except Exception as e:
                print(f"Error reading directory {base_dir}: {e}")
        else:
            print(f"\nDirectory does not exist: {base_dir}")

if __name__ == "__main__":
    check_specific_model()