# Methodology: Model-Corrected Labels

## The Problem with Raw arXiv Labels

arXiv categories are **author-declared**, not peer-reviewed. Our analysis found:
- **3.5% mislabeling rate** across CS categories
- Authors choose categories at submission time without validation
- Multi-label assignments often incomplete or incorrect

**Example mislabeling:**
```
Paper: "Deep Learning for Database Query Optimization"
Author label: cs.DB only
Correct labels: cs.DB + cs.LG (uses deep learning)
```

## Our Innovation: Model-Corrected Labels

Instead of training on noisy author labels, we use a two-stage process:

### Stage 1: High-Quality Seed Model

Train initial classifier on **manually validated subset**:
- 1,000 papers per category (8,000 total)
- Manual verification by domain experts
- Conservative labeling (high precision)

**Result:** 92% F1 seed model on clean data

### Stage 2: Label Correction at Scale

Use seed model to **correct** training labels for full dataset:

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

**Merging strategy:**
- Keep all author-declared labels (respect author intent)
- Add high-confidence model predictions (threshold > 0.8)
- Remove low-confidence author labels (if model strongly disagrees)

### Stage 3: Retrain on Corrected Labels

Train final classifier on corrected dataset:
- 31,128 papers with improved labels
- GraphSAGE + citation network features
- SciBERT embeddings

**Result:** 94.26% F1 (15-25pp improvement over baseline)

## Why This Works

### 1. Leverages Human Knowledge
- Respects author expertise (keeps declared labels)
- Adds systematic improvements (model corrections)

### 2. Scalable Quality Improvement
- Manual validation: 1,000 papers (feasible)
- Automated correction: 31,128 papers (scalable)

### 3. Compound Learning Effect
- Better labels -> better model -> better labels
- Can iterate for further improvement

## Experimental Validation

We validated effectiveness by comparing:

| Approach | Average F1 | Training Data |
|----------|-----------|---------------|
| Baseline (raw labels) | 75.3% | Author-declared categories |
| Manual-only | 92.1% | 8,000 manually validated papers |
| Model-corrected | **94.26%** | 31,128 corrected papers |

**Key finding:** Model-corrected labels on large dataset outperforms small manually-curated dataset.

## Label Correction Examples

### Example 1: Missing Multi-Label

**Paper:** "BERT for Code Search and Documentation"

```
Author labels:    [cs.SE]
Model prediction: [cs.SE: 0.95, cs.CL: 0.87, cs.AI: 0.82]
Corrected labels: [cs.SE, cs.CL, cs.AI]
```

**Why:** Paper uses NLP (cs.CL) and transformers (cs.AI) for software engineering task

### Example 2: Category Refinement

**Paper:** "Distributed Training of Neural Networks"

```
Author labels:    [cs.AI, cs.DC]
Model prediction: [cs.LG: 0.91, cs.DC: 0.88, cs.AI: 0.45]
Corrected labels: [cs.LG, cs.DC]
```

**Why:** Paper is about ML systems (cs.LG) not general AI (cs.AI)

### Example 3: Respect Author Intent

**Paper:** "Database Systems for Machine Learning"

```
Author labels:    [cs.DB, cs.LG]
Model prediction: [cs.DB: 0.93, cs.LG: 0.62, cs.AI: 0.81]
Corrected labels: [cs.DB, cs.LG, cs.AI]
```

**Why:** Keep author labels (cs.LG despite lower confidence), add cs.AI

## Implementation Details

### Confidence Calibration

Model predictions are calibrated using temperature scaling:

```python
def calibrate_prediction(logits, temperature=1.5):
    """Calibrate model confidence"""
    return sigmoid(logits / temperature)
```

**Effect:** Prevents over-confident predictions from overwhelming author labels

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

This technique is applicable to any multi-label classification with:
1. Noisy crowd-sourced labels (e.g., Stack Overflow tags)
2. Author-declared categories (e.g., research papers, patents)
3. User-generated content (e.g., social media hashtags)

**Requirements:**
- Ability to manually validate small seed set (500-1000 samples)
- Moderate labeling noise (3-10% error rate)
- Multi-label structure (harder for single-label problems)

## Limitations

1. **Requires seed data:** Need clean subset for initial model
2. **Computational cost:** Two-stage training process
3. **Risk of bias amplification:** Model biases can propagate to corrected labels
4. **Domain expertise needed:** Manual validation requires subject matter experts

## Future Work

- **Active learning:** Identify most impactful corrections
- **Uncertainty quantification:** Flag low-confidence corrections for review
- **Iterative refinement:** Multiple correction rounds
- **Cross-validation:** Use multiple seed models for ensemble correction

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
