# LLaMA Connectivity Test

A tool for testing LLaMA model connectivity and basic functionality.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/llama-connectivity-test.git
cd llama-connectivity-test
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
- Copy `.env.example` to `.env`
- Update `LLAMA_MODEL_PATH` with your model path

## Usage

Run the connectivity test:
```bash
python src/run_test.py
```

## Test Components

1. **System Check**
   - Verifies RAM and GPU requirements
   - Checks available resources

2. **Model Path Verification**
   - Confirms model files exist
   - Validates required components

3. **Model Loading**
   - Tests tokenizer loading
   - Tests model loading

4. **Inference Test**
   - Performs basic inference
   - Verifies model output

## Logs

Logs are stored in `logs/llama_test.log`

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- 16GB+ RAM
- GPU recommended (8GB+ VRAM)