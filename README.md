# arXiv Multi-Label Classification

Multi-label classifier for computer science papers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![F1 Score](https://img.shields.io/badge/F1-94.26%25-brightgreen.svg)](https://github.com/green8-dot/arxiv-multilabel-classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/transformers/)
[![arXiv](https://img.shields.io/badge/arXiv-Data-b31b1b.svg)](https://arxiv.org/)

## Overview

Multi-label classification for arXiv computer science papers. 94.26% average F1 across 8 categories. Uses GraphSAGE and model-corrected labels.

## Performance Metrics

| Category | F1 Score | Improvement over Baseline |
|----------|----------|---------------------------|
| cs.AI (Artificial Intelligence) | 97.45% | +18pp |
| cs.CL (Computation & Language) | 98.04% | +22pp |
| cs.CV (Computer Vision) | 95.41% | +19pp |
| cs.DB (Databases) | 95.73% | +21pp |
| cs.DC (Distributed Computing) | 90.58% | +15pp |
| cs.LG (Machine Learning) | 89.82% | +17pp |
| cs.RO (Robotics) | 88.48% | +16pp |
| cs.SE (Software Engineering) | 98.58% | +25pp |
| **Average** | **94.26%** | **+19pp** |

Target was 80% F1.

## Model-Corrected Labels

arXiv categories contain 3.5% mislabeling (author-declared, not validated). This system uses a trained model to correct training labels before final training. Result: 15-25pp F1 improvement over baseline.

## Architecture

- **Base Model:** SciBERT (scientific domain pre-trained BERT)
- **Graph Component:** GraphSAGE (citation + co-authorship networks)
- **Training:** Mixed precision (fp16), focal loss for class imbalance
- **Framework:** PyTorch + HuggingFace Transformers

## Dataset

- **31,128 papers** from arXiv CS categories
- **Citation network:** 99.7% author recovery rate
- **Multi-label:** Papers can belong to multiple categories
- **Data source:** Public arXiv metadata

## Installation

```bash
# Clone repository
git clone https://github.com/green8-dot/arxiv-multilabel-classification.git
cd arxiv-multilabel-classification

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
python download_models.py
```

## Usage

### Train Your Own Model

```python
from train import train_classifier

# Train classifier for cs.AI
train_classifier(
    category='cs.AI',
    data_path='data/papers.json',
    epochs=3,
    batch_size=32
)
```

### Use Pre-Trained Models

```python
from inference import classify_paper

# Classify a paper
result = classify_paper(
    title="Attention Is All You Need",
    abstract="...",
    model='cs.AI'
)

print(f"Predicted categories: {result.categories}")
print(f"Confidence: {result.confidence:.2%}")
```

## Repository Structure

```
arxiv-multilabel-classification/
├── models/                  # Pre-trained classifiers
├── training/
│   ├── train.py            # Training scripts
│   ├── evaluate.py         # Evaluation tools
│   └── configs/            # Model configurations
├── inference/
│   ├── classify.py         # Inference API
│   └── batch_process.py    # Batch classification
├── data/
│   ├── preprocessing.py    # Data preparation
│   └── download.py         # Dataset fetcher
├── docs/
│   ├── METHODOLOGY.md      # Model-corrected labels technique
│   ├── PERFORMANCE.md      # Detailed benchmarks
│   └── ARCHITECTURE.md     # System design
└── requirements.txt
```

## Technical Details

**Machine Learning:**
- Multi-label classification with focal loss
- Graph neural networks (GraphSAGE)
- SciBERT fine-tuning
- Mixed precision training (fp16)
- Model-corrected label generation

**Engineering:**
- Docker deployment
- REST API for inference
- Batch processing

## Results

- 8/8 categories above 85% F1
- Average 94.26% F1 (target: 80%)
- Model-corrected labels improved all categories by 15-25pp

**Findings:**
1. Model-corrected labels improve noisy datasets
2. Citation networks add 5-8pp improvement
3. Focal loss reduces class imbalance issues
4. SciBERT outperforms generic BERT by 12pp

## Applications

Techniques applicable to:
- Scientific document classification
- Multi-label text categorization
- Knowledge graph integration
- Noisy label correction

## Future Work

- [ ] Extend to additional arXiv categories
- [ ] Multilingual support
- [ ] Real-time classification API
- [ ] Explainable AI integration
- [ ] Active learning for label quality

## Citation

If you use this work, please cite:

```bibtex
@software{arxiv_multilabel_2025,
  title={arXiv Multi-Label Classification with Model-Corrected Labels},
  author={OrbitScope Research},
  year={2025},
  url={https://github.com/green8-dot/arxiv-multilabel-classification}
}
```

## License

MIT License - See LICENSE file for details

## Author

OrbitScope Research - ML Engineering & Research
- GitHub: [github.com/green8-dot](https://github.com/green8-dot)
- Website: https://orbitscope.io/
## Acknowledgments

- arXiv for public dataset access
- SciBERT authors for pre-trained models
- PyTorch + HuggingFace communities

---

**Last Updated:** 2025-10-22

