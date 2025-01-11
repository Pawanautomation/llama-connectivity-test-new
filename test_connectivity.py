# test_connectivity.py
from huggingface_hub import login, whoami
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_huggingface_connection():
    try:
        # Load environment variables
        load_dotenv()
        token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not token:
            logger.error("❌ No HUGGINGFACE_TOKEN found in .env file")
            return False
            
        # Try to log in
        logger.info("Attempting to log in to Hugging Face...")
        login(token=token)
        
        # Verify login
        user_info = whoami()
        logger.info(f"✅ Successfully logged in as: {user_info['name']}")
        
        # Check LLaMA access
        logger.info("Checking access to LLaMA model...")
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            # Try to get model info
            model_info = api.model_info("meta-llama/Llama-2-7b-chat-hf")
            logger.info(f"✅ Have access to LLaMA model: {model_info.modelId}")
        except Exception as e:
            logger.error(f"❌ No access to LLaMA model: {str(e)}")
            logger.info("Please accept the model terms at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error during connection test: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n=== Hugging Face Connection Test ===\n")
    success = test_huggingface_connection()
    print("\n=== Test Results ===")
    print("Connection Test:", "✅ Passed" if success else "❌ Failed")