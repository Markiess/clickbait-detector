# Clickbait Detector

A lightweight clickbait detector built via **LLM knowledge distillation** — a large language model (teacher) is used to generate soft labels for headlines, and a compact student model is trained on those labels to produce a fast, accurate binary classifier.

---

## Table of Contents

1. [Overview](#overview)
2. [Experiment Setup](#experiment-setup)
3. [Pipeline](#pipeline)
4. [Results](#results)
5. [Usage](#usage)
6. [Requirements](#requirements)

---

## Overview

Clickbait headlines are designed to attract clicks through sensationalism rather than informative content. This project explores using **knowledge distillation from a large language model (LLM)** to train a compact classifier that can detect clickbait headlines with high accuracy at low inference cost.

The key idea:
- A powerful LLM (teacher) scores each headline with a soft probability of being clickbait.
- A smaller transformer-based model (student) is fine-tuned on those soft labels using a KL-divergence distillation loss.
- The resulting student model is ~10× smaller than the teacher and runs significantly faster.

---

## Experiment Setup

### Dataset

| Source | Split | Size |
|--------|-------|------|
| [WebIS Clickbait Corpus 2017](https://webis.de/data/webis-clickbait-17.html) | Train | 19,538 |
| WebIS Clickbait Corpus 2017 | Validation | 2,500 |
| WebIS Clickbait Corpus 2017 | Test | 4,551 |

Each sample consists of a **tweet text** (headline + optional teaser) paired with a human-annotated clickbait score in `[0, 1]`. Scores ≥ 0.5 are treated as clickbait (positive class).

Class distribution in training set:
- **Clickbait**: ~50 %
- **Non-clickbait**: ~50 %

### Teacher Model

| Property | Value |
|----------|-------|
| Model | `gpt-3.5-turbo` (OpenAI API) |
| Prompt style | Zero-shot chain-of-thought |
| Output | Soft probability `P(clickbait)` ∈ [0, 1] |
| Temperature | 0.0 (deterministic) |

A zero-shot prompt was used to ask the LLM to rate each headline on a 0–1 scale:

```
You are a media-literacy expert. Given the following headline, output a single decimal number between 0 and 1 indicating how likely it is to be clickbait (0 = definitely not clickbait, 1 = definitely clickbait). Respond with only the number.

Headline: "{headline}"
```

### Student Model

| Property | Value |
|----------|-------|
| Base model | `distilbert-base-uncased` |
| Max sequence length | 128 tokens |
| Classification head | Single linear layer → sigmoid |
| Training objective | αBCE(hard labels) + (1−α)KL(soft labels), α = 0.3 |

### Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.1 |
| Transformers (HuggingFace) | 4.38 |
| OpenAI SDK | 1.14 |
| Hardware | NVIDIA T4 GPU (Google Colab) |

---

## Pipeline

```
Raw Dataset
    │
    ▼
1. Data Preprocessing
   • Lowercase, strip URLs & HTML entities
   • Tokenise with DistilBERT tokeniser (max_len=128)
   • Train/val/test split (stratified by label)
    │
    ▼
2. Teacher Labelling (LLM)
   • Call gpt-3.5-turbo for every training headline
   • Parse soft probability from response
   • Cache results to avoid repeated API calls
    │
    ▼
3. Student Model Training (Distillation)
   • Combined loss: α·BCE(hard) + (1−α)·KL(teacher soft labels)
   • Optimiser: AdamW, lr=2e-5, weight decay=0.01
   • Scheduler: linear warmup (10 % steps) → cosine decay
   • Epochs: 5  |  Batch size: 32
   • Best checkpoint selected by validation F1
    │
    ▼
4. Evaluation
   • Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
   • Baselines: TF-IDF + Logistic Regression, fine-tuned DistilBERT (hard labels only)
```

### Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Dataset statistics, class balance, headline length distributions |
| `02_teacher_labelling.ipynb` | LLM prompting, soft label generation, label agreement analysis |
| `03_student_training.ipynb` | Distillation training loop, loss curves, checkpoint selection |
| `04_evaluation.ipynb` | Metric computation, confusion matrix, error analysis |

---

## Results

### Model Comparison on Test Set

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| TF-IDF + Logistic Regression (baseline) | 0.781 | 0.773 | 0.792 | 0.782 | 0.861 |
| DistilBERT (hard labels only) | 0.847 | 0.839 | 0.861 | 0.850 | 0.923 |
| **DistilBERT + LLM Distillation (ours)** | **0.871** | **0.864** | **0.882** | **0.873** | **0.941** |

> The distillation objective provides a consistent **+2.3 F1** improvement over fine-tuning on hard labels alone, demonstrating that teacher soft labels carry meaningful calibration signal.

### Training Curves

- Validation loss stabilises around epoch 3; F1 peaks at epoch 4.
- Distillation loss (KL component) decreases monotonically, confirming the student converges toward the teacher's distribution.

### Inference Speed

| Model | Latency (CPU, ms) | Model Size |
|-------|--------------------|------------|
| gpt-3.5-turbo (teacher) | ~800 ms / sample | — |
| DistilBERT student | ~12 ms / sample | 66 M params |

The student model is **~67× faster** at inference than querying the teacher API.

### Error Analysis

Common misclassification patterns:
- Headlines with **numbers/statistics** ("10 things you didn't know…") are occasionally missed — they overlap stylistically with legitimate listicle journalism.
- Very short or ambiguous headlines (< 5 tokens) are harder to classify for both teacher and student.

---

## Usage

```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./checkpoints/best")
model.eval()

headline = "You won't believe what happened next!"
inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    prob_clickbait = torch.softmax(logits, dim=-1)[:, 1].item()

print(f"Clickbait probability: {prob_clickbait:.3f}")
```

---

## Requirements

```
openai>=1.14
transformers>=4.38
torch>=2.1
datasets
scikit-learn
pandas
matplotlib
seaborn
tqdm
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this work, please cite the WebIS Clickbait Corpus:

```
@inproceedings{potthast:2017,
  author    = {Martin Potthast and Sebastian Köpsel and Benno Stein and Matthias Hagen},
  title     = {Clickbait Detection},
  booktitle = {Advances in Information Retrieval},
  year      = {2017},
  publisher = {Springer},
}
```
