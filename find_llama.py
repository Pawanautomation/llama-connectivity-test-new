# find_llama_direct.py
import os
from pathlib import Path

def find_llama_models():
    # Use the cache directory we found
    cache_dir = Path(r"C:\Users\pawan\.cache\huggingface\hub")
    
    print(f"\nSearching for LLaMA models in: {cache_dir}")
    print("-" * 50)
    
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    # Walk through the directory
    found_models = False
    for root, dirs, files in os.walk(cache_dir):
        if "llama" in root.lower():
            found_models = True
            print(f"\nFound potential LLaMA files in:")
            print(f"Path: {root}")
            print("\nContents:")
            for item in os.listdir(root):
                print(f"- {item}")

    if not found_models:
        print("\nNo LLaMA models found in cache directory")
        # Also check models directory specifically
        models_dir = cache_dir.parent / "transformers"
        if models_dir.exists():
            print(f"\nChecking transformers directory: {models_dir}")
            for item in os.listdir(models_dir):
                if "llama" in item.lower():
                    print(f"Found: {item}")
                    print(f"Full path: {models_dir / item}")

if __name__ == "__main__":
    find_llama_models()