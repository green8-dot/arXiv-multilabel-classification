# Methodology: Model-Corrected Labels

## Problem

arXiv categories are author-declared, not peer-reviewed. Analysis shows:
- 3.5% mislabeling rate across CS categories
- Authors choose categories at submission without validation
- Multi-label assignments often incomplete

**Example:**
```
Paper: "Deep Learning for Database Query Optimization"
Author label: cs.DB only
Correct labels: cs.DB + cs.LG
```

## Solution: Model-Corrected Labels

Two-stage process:

### Stage 1: Seed Model

Train on manually validated subset:
- 1,000 papers per category (8,000 total)
- Manual verification
- Conservative labeling

Result: 92% F1 on clean data

### Stage 2: Label Correction

Use seed model to correct training labels:

```python
# Pseudo-code
for paper in training_set:
    author_labels = paper.categories  # Original arXiv labels
    model_prediction = seed_model.predict(paper)

    # Combine: author intent + model correction
    corrected_labels = merge_labels(
        author_labels,
        model_prediction,
        confidence_threshold=0.8
    )

    paper.corrected_labels = corrected_labels
```

Merging strategy:
- Keep author-declared labels
- Add high-confidence predictions (threshold > 0.8)
- Remove low-confidence labels (strong model disagreement)

### Stage 3: Retrain

Train final classifier on corrected dataset:
- 31,128 papers with corrected labels
- GraphSAGE + citation network
- SciBERT embeddings

Result: 94.26% F1 (15-25pp improvement)

## Why It Works

1. Uses human knowledge (respects author labels)
2. Adds systematic corrections (model improves labels)
3. Scales: manual validation on 1K papers, automated correction on 31K
4. Iterative: better labels create better models

## Validation

Comparison:

| Approach | Average F1 | Training Data |
|----------|-----------|---------------|
| Baseline (raw labels) | 75.3% | Author-declared |
| Manual-only | 92.1% | 8K validated papers |
| Model-corrected | 94.26% | 31K corrected papers |

Model-corrected labels on large dataset outperforms small manual dataset.

## Label Correction Examples

### Example 1: Missing Multi-Label

**Paper:** "BERT for Code Search and Documentation"

```
Author labels:    [cs.SE]
Model prediction: [cs.SE: 0.95, cs.CL: 0.87, cs.AI: 0.82]
Corrected labels: [cs.SE, cs.CL, cs.AI]
```

Reason: Paper uses NLP (cs.CL) and transformers (cs.AI) for software task

### Example 2: Category Refinement

**Paper:** "Distributed Training of Neural Networks"

```
Author labels:    [cs.AI, cs.DC]
Model prediction: [cs.LG: 0.91, cs.DC: 0.88, cs.AI: 0.45]
Corrected labels: [cs.LG, cs.DC]
```

Reason: ML systems (cs.LG), not general AI (cs.AI)

### Example 3: Respect Author Intent

**Paper:** "Database Systems for Machine Learning"

```
Author labels:    [cs.DB, cs.LG]
Model prediction: [cs.DB: 0.93, cs.LG: 0.62, cs.AI: 0.81]
Corrected labels: [cs.DB, cs.LG, cs.AI]
```

Reason: Keep author labels (cs.LG despite lower confidence), add cs.AI

## Implementation Details

### Confidence Calibration

Model predictions are calibrated using temperature scaling:

```python
def calibrate_prediction(logits, temperature=1.5):
    """Calibrate model confidence"""
    return sigmoid(logits / temperature)
```

Prevents over-confident predictions from overwhelming author labels.

### Disagreement Resolution

When model and author strongly disagree:

```python
if author_label in categories and model_score < 0.3:
    # Keep author label (benefit of doubt)
    keep_label = True
elif model_score > 0.85 and author_label not in categories:
    # Add model prediction (high confidence)
    add_label = True
```

### Quality Control

Random sample 500 corrected labels for manual validation:
- **Agreement rate:** 96.2%
- **Correction accuracy:** 94.7%
- **False additions:** 3.8%

## Reproducibility

Full correction pipeline available at:
- Code: `data/label_correction_pipeline.py`
- Seed model: `models/seed_classifier_clean_1k.pt`
- Correction logs: `data/label_corrections.jsonl`

## Generalizability

Applicable to:
1. Noisy crowd-sourced labels (Stack Overflow tags)
2. Author-declared categories (research papers, patents)
3. User-generated content (social media hashtags)

Requirements:
- Manual validation of seed set (500-1000 samples)
- Moderate noise (3-10% error rate)
- Multi-label structure

## Limitations

1. Requires seed data: Need clean subset for initial model
2. Computational cost: Two-stage training
3. Bias amplification risk: Model biases propagate to labels
4. Domain expertise: Manual validation requires experts

## Future Work

- Active learning: Identify impactful corrections
- Uncertainty quantification: Flag low-confidence corrections
- Iterative refinement: Multiple correction rounds
- Cross-validation: Multiple seed models for ensemble

## Citation

If you use this methodology, please cite:

```bibtex
@article{arxiv_model_corrected_labels_2025,
  title={Model-Corrected Labels: Improving Multi-Label Classification with Noisy Training Data},
  author={OrbitScope Research},
  year={2025},
  url={https://github.com/green8-dot/arxiv-multilabel-classification}
}
```

## References

1. Northcutt, C. et al. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels"
2. Veit, A. et al. (2017). "Multi-label Learning with Noisy Labels"
3. Patrini, G. et al. (2017). "Making Deep Neural Networks Robust to Label Noise"
