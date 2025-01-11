# test_staged_loading.py
import os
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from dotenv import load_dotenv
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StagedModelTest:
    def __init__(self):
        """Initialize the staged model test environment"""
        load_dotenv()
        self.model_path = os.getenv("LLAMA_MODEL_PATH")
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

        # Validate environment
        if not self.model_path:
            raise ValueError("LLAMA_MODEL_PATH not set in environment")
        if not self.token:
            raise ValueError("HUGGINGFACE_TOKEN not set in environment")

    def log_memory(self, stage: str) -> None:
        """Log memory usage at a given stage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"\nMemory usage at {stage}:")
        logger.info(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
        logger.info(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB"
            )

    def stage1_system_check(self) -> bool:
        """Check system requirements"""
        try:
            logger.info("\n=== Stage 1: System Check ===")

            # Check Python and PyTorch versions
            logger.info(f"PyTorch Version: {torch.__version__}")

            # Check CPU
            cpu_count = psutil.cpu_count()
            logger.info(f"CPU Cores: {cpu_count}")

            # Check RAM
            memory = psutil.virtual_memory()
            logger.info(f"Total RAM: {memory.total / (1024**3):.2f} GB")
            logger.info(
                f"Available RAM: {memory.available / (1024**3):.2f} GB"
            )

            # Check GPU if available
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(
                    0
                ).total_memory / 1e9
                logger.info(f"GPU: {gpu_name}")
                logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
            else:
                logger.info("No GPU available, using CPU")

            # Log memory baseline
            self.log_memory("system check")
            return True

        except Exception as e:
            logger.error(f"System check failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False

    def stage2_load_tokenizer(self) -> bool:
        """Load and configure the tokenizer"""
        try:
            logger.info("\n=== Stage 2: Loading Tokenizer ===")

            # Load tokenizer with safety checks
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                token=self.token,
                trust_remote_code=True,
            )

            # Configure padding token
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Verify tokenizer configuration
            logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
            logger.info(f"Pad token: {self.tokenizer.pad_token}")
            logger.info(f"EOS token: {self.tokenizer.eos_token}")

            logger.info("✅ Tokenizer loaded successfully")
            self.log_memory("tokenizer loading")
            return True

        except Exception as e:
            logger.error(f"Tokenizer loading failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False

    def stage3_load_model(self) -> bool:
        """Load the LLaMA model"""
        try:
            logger.info("\n=== Stage 3: Loading Model ===")
            logger.info("Starting model load (this may take a while)...")

            # Configure model loading parameters
            model_kwargs = {
                "token": self.token,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs,
            )

            # Move model to appropriate device
            if self.device == "cpu":
                self.model = self.model.to(self.device)

            logger.info("✅ Model loaded successfully")
            self.log_memory("model loading")
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False

    def stage4_test_inference(self) -> bool:
        """Test inference with a simpler prompt for CPU-based testing"""
        try:
            logger.info("\n=== Stage 4: Testing Inference ===")

            # Simpler test prompt
            test_text = "What are three key principles of DevOps?"

            logger.info(f"Test input: {test_text}")

            # Reduced generation parameters
            generation_config = {
                "max_new_tokens": 100,  # Significantly reduced
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_beams": 1,  # Disable beam search for faster inference
            }

            # Tokenize input
            logger.info("Tokenizing input...")
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=100,  # Reduced max length
            )

            if self.device == "cpu":
                inputs = inputs.to(self.model.device)

            # Generate response
            logger.info("Generating response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config,
                )

            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            logger.info(f"\nModel response: {response}")

            self.log_memory("inference")
            return True

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False

    def cleanup(self) -> None:
        """Clean up resources and free memory"""
        try:
            logger.info("\n=== Cleanup ===")

            # Remove model from memory
            if self.model is not None:
                del self.model

            # Remove tokenizer from memory
            if self.tokenizer is not None:
                del self.tokenizer

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)


def main():
    """Main execution function"""
    tester = StagedModelTest()

    # Define test stages
    stages: List[Tuple[str, callable]] = [
        ("System Check", tester.stage1_system_check),
        ("Load Tokenizer", tester.stage2_load_tokenizer),
        ("Load Model", tester.stage3_load_model),
        ("Test Inference", tester.stage4_test_inference),
    ]

    # Execute stages
    results: Dict[str, bool] = {}
    try:
        for stage_name, stage_func in stages:
            logger.info(f"\nExecuting: {stage_name}")
            results[stage_name] = stage_func()

            if not results[stage_name]:
                logger.error(f"❌ {stage_name} failed. Stopping test.")
                break

    finally:
        tester.cleanup()

    # Report results
    logger.info("\n=== Test Results ===")
    for stage_name, passed in results.items():
        status = "✅ Passed" if passed else "❌ Failed"
        logger.info(f"{stage_name}: {status}")


if __name__ == "__main__":
    main()
