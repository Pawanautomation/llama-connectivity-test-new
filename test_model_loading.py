# test_model_loading.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    try:
        # Load environment variables
        load_dotenv()
        model_path = os.getenv('LLAMA_MODEL_PATH')
        logger.info(f"Testing model loading from: {model_path}")

        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Try loading tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Tokenizer loaded successfully")

        # Try loading model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(device)
        logger.info("✅ Model loaded successfully")

        # Try a test inference
        logger.info("Testing inference...")
        test_input = "What is DevOps?"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("✅ Test inference successful")
        logger.info(f"Test response: {response}")

        return True

    except Exception as e:
        logger.error(f"Error during model test: {str(e)}")
        return False

    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("\n=== LLaMA Model Loading Test ===\n")
    success = test_model_loading()
    print("\n=== Test Results ===")
    print("Model Loading:", "✅ Passed" if success else "❌ Failed")