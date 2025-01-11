# src/llama_test/core.py
import logging
import sys
import torch
import psutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.settings import (
    LLAMA_MODEL_PATH,
    LOG_LEVEL,
    LOG_FILE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TEMPERATURE,
    MIN_RAM_GB,
    MIN_GPU_MEMORY_GB
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

class LlamaConnectivityTest:
    """Test LLaMA model connectivity and basic functionality."""
    
    def __init__(self):
        self.model_path = Path(LLAMA_MODEL_PATH)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized LlamaConnectivityTest with device: {self.device}")

    def run_all_tests(self) -> dict:
        """Run all connectivity tests and return results."""
        results = {
            'system_check': False,
            'model_path': False,
            'model_load': False,
            'inference': False
        }
        
        try:
            # Run tests sequentially
            results['system_check'] = self.check_system_resources()
            if results['system_check']:
                results['model_path'] = self.verify_model_path()
                if results['model_path']:
                    results['model_load'] = self.load_model()
                    if results['model_load']:
                        results['inference'] = self.test_inference()
        
        except Exception as e:
            logger.error(f"Error during tests: {str(e)}")
        
        finally:
            self.cleanup()
            
        return results

    def check_system_resources(self) -> bool:
        """Check if system meets minimum requirements."""
        try:
            # Check RAM
            ram_gb = psutil.virtual_memory().total / 1e9
            logger.info(f"Available RAM: {ram_gb:.2f} GB")
            
            if ram_gb < MIN_RAM_GB:
                logger.error(f"Insufficient RAM. Required: {MIN_RAM_GB}GB, Available: {ram_gb:.2f}GB")
                return False

            # Check GPU if available
            if self.device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")
                
                if gpu_memory < MIN_GPU_MEMORY_GB:
                    logger.error(f"Insufficient GPU memory. Required: {MIN_GPU_MEMORY_GB}GB, Available: {gpu_memory:.2f}GB")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return False

    def verify_model_path(self) -> bool:
        """Verify if model path exists and contains required files."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model path does not exist: {self.model_path}")
                return False

            required_files = ['config.json', 'tokenizer.json']
            for file in required_files:
                if not (self.model_path / file).exists():
                    logger.error(f"Missing required file: {file}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error verifying model path: {str(e)}")
            return False

    def load_model(self) -> bool:
        """Load the LLaMA model and tokenizer."""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.to(self.device)
            
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def test_inference(self, test_prompt: str = "What is DevOps?") -> bool:
        """Test if model can perform inference."""
        try:
            logger.info("Testing inference...")
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=DEFAULT_MAX_LENGTH,
                    temperature=DEFAULT_TEMPERATURE,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Test inference successful. Response: {response[:100]}...")
            return True

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return False

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")