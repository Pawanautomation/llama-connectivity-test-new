# test_env.py
from dotenv import load_dotenv
import os

def test_env_variables():
    # Load .env file
    load_dotenv()
    
    # Get variables
    model_path = os.getenv('LLAMA_MODEL_PATH')
    token = os.getenv('HUGGINGFACE_TOKEN')
    log_level = os.getenv('LOG_LEVEL')
    
    print("\n=== Environment Variables Test ===")
    print(f"\nLLAMA_MODEL_PATH: {model_path}")
    print(f"HUGGINGFACE_TOKEN: {'[Set]' if token else '[Not Set]'}")
    print(f"LOG_LEVEL: {log_level}")
    
    # Check if variables are properly set
    if not model_path:
        print("\n❌ LLAMA_MODEL_PATH is not set")
    if not token:
        print("❌ HUGGINGFACE_TOKEN is not set")
    if not log_level:
        print("❌ LOG_LEVEL is not set")
    
    if model_path and token and log_level:
        print("\n✅ All required environment variables are set")

if __name__ == "__main__":
    test_env_variables()