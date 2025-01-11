# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# LLaMA model settings
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH')

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'logs' / 'llama_test.log'

# Model settings
DEFAULT_MAX_LENGTH = 100
DEFAULT_TEMPERATURE = 0.7

# System requirements
MIN_RAM_GB = 16
MIN_GPU_MEMORY_GB = 8