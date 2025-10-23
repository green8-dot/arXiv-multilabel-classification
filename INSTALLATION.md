# Installation Guide

## System Requirements

- **Python:** 3.8 or higher
- **CUDA:** 11.0+ (for GPU acceleration, optional)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 5GB for models and data

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/green8-dot/arxiv-multilabel-classification.git
cd arxiv-multilabel-classification
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## GPU Setup (Optional but Recommended)

### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Install PyTorch with CUDA Support

If CUDA is not detected, reinstall PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Download Pre-Trained Models

### Option 1: Automatic

```bash
python download_models.py
```

Downloads all 8 classifiers (~2GB).

### Option 2: Manual

Download from [releases page](https://github.com/green8-dot/arxiv-multilabel-classification/releases) and place in `models/`:

```
models/
├── cs_ai_classifier.pt
├── cs_cl_classifier.pt
├── cs_cv_classifier.pt
├── cs_db_classifier.pt
├── cs_dc_classifier.pt
├── cs_lg_classifier.pt
├── cs_ro_classifier.pt
└── cs_se_classifier.pt
```

## Neo4j Setup (Optional)

System works without Neo4j. Graph features add 5-8pp accuracy improvement.

### Install Neo4j

```bash
# Ubuntu/Debian
sudo apt install neo4j

# macOS
brew install neo4j

# Or use Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### Configure Connection

Create `config.json`:

```json
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "your_password"
  }
}
```

### Import Citation Network Data

```bash
python data/import_to_neo4j.py --data-path data/papers.json
```

## Troubleshooting

### Issue: Out of Memory Errors

**Solution:** Reduce batch size in configuration

```python
# In your training/inference code
config = {
    'batch_size': 16  # Default is 32, reduce to 16 or 8
}
```

### Issue: CUDA Out of Memory

**Solution:** Use mixed precision training

```python
config = {
    'mixed_precision': True,  # Reduces GPU memory by 50%
    'batch_size': 16
}
```

### Issue: Slow Training

**Solutions:**
1. Enable GPU acceleration (see GPU Setup above)
2. Use pre-trained models instead of training from scratch
3. Reduce dataset size for testing

### Issue: Import Errors

**Solution:** Ensure virtual environment is activated

```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

## Environment Variables

Optional configuration:

```bash
export ARXIV_DATA_PATH="/path/to/data"       # Custom data directory
export ARXIV_MODEL_PATH="/path/to/models"    # Custom model directory
export ARXIV_CACHE_DIR="/path/to/cache"      # Cache directory
export ARXIV_LOG_LEVEL="INFO"                # Logging level
```

## Testing Installation

Run the test suite to verify everything is working:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_classification.py::test_model_loading PASSED
tests/test_classification.py::test_inference PASSED
tests/test_training.py::test_data_loading PASSED
================================ 3 passed in 2.45s ================================
```

## Next Steps

- Read [METHODOLOGY.md](METHODOLOGY.md) to understand the model-corrected labels technique
- See [README.md](README.md) for usage examples
- Check [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design details

## Support

If you encounter issues:
1. Check [GitHub Issues](https://github.com/green8-dot/arxiv-multilabel-classification/issues)
2. Review troubleshooting section above
3. Open a new issue with error details and system info

## System Info for Bug Reports

When reporting issues, include:

```bash
python --version
pip freeze > requirements_installed.txt
nvidia-smi  # If using GPU
uname -a    # Linux/macOS
```
