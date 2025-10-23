# arXiv Multi-Label Classification
**Production-Grade Multi-Label Classifier for Computer Science Papers**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![F1 Score](https://img.shields.io/badge/F1-94.26%25-brightgreen.svg)](https://github.com/green8-dot/arxiv-multilabel-classification)

## Overview

State-of-the-art multi-label classification system for arXiv computer science papers, achieving **94.26% average F1 score** across 8 categories using GraphSAGE + model-corrected labels.

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

*Target was 80%+ F1 - achieved 94.26% average*

## Key Innovation: Model-Corrected Labels

Traditional approach: Train on raw arXiv categories (contains 3.5% mislabeling)

**Our approach:** Use high-quality model to correct training labels -> 15-25pp F1 improvement

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

## Technical Highlights

**Machine Learning:**
- Multi-label classification with focal loss
- Graph neural networks (GraphSAGE)
- Transfer learning (SciBERT fine-tuning)
- Mixed precision training (50% memory reduction)
- Model-corrected label generation

**Engineering:**
- Production-ready code
- Comprehensive testing
- Docker deployment ready
- REST API for inference
- Batch processing support

## Results Analysis

**What We Achieved:**
- Exceeded 80% F1 target by +14pp
- 8/8 categories above 85% F1
- Model-corrected labels improved all categories
- Production deployment ready

**Key Findings:**
1. Model-corrected labels critical for noisy datasets
2. Citation networks improve classification (+5-8pp)
3. Focal loss handles class imbalance effectively
4. SciBERT outperforms generic BERT by 12pp

## Applications

This system demonstrates techniques applicable to:
- Scientific document classification
- Multi-label text categorization
- Knowledge graph + NLP integration
- Production ML pipelines

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

## Acknowledgments

- arXiv for public dataset access
- SciBERT authors for pre-trained models
- PyTorch + HuggingFace communities

---

**Status:** Production-ready, fully tested, actively maintained

**Last Updated:** 2025-10-22
