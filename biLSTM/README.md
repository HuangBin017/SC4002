# Joint BiLSTM Experiments

Both notebooks use the official MASSIVE 1.0 `en-US` splits and a single shared model for intent classification and exact-span slot filling. GloVe 6B 300D embeddings are fine-tuned during training.

## Methods

- **Baseline:** one-layer BiLSTM with 128 hidden units per direction, masked mean pooling for the intent head, and an independent token-level BIO slot classifier. Total parameters: **2,072,467**.
- **Enhanced:** two-layer BiLSTM with 256 hidden units per direction, character-CNN features, attention/mean/max intent pooling, predicted-intent conditioning, and a structured boundary CRF with span-consistent slot typing. Total parameters: **5,278,513**.

## Results

| Model | Split | Intent accuracy | Slot micro-F1 | Combined score |
|---|---|---:|---:|---:|
| Baseline | Validation | 85.93% | 75.06% | 80.50% |
| Baseline | Test | 84.57% | 71.84% | 78.20% |
| Enhanced | Validation | **87.41%** | **80.95%** | **84.18%** |
| Enhanced | Test | **86.68%** | **78.09%** | **82.39%** |

The enhanced system improved test intent accuracy by **2.12 points** and exact-span slot micro-F1 by **6.25 points** over the baseline.

## Notebooks

- `baseline/main.ipynb` - executed GloVe-300D baseline.
- `suggestions/main.ipynb` - executed enhanced joint BiLSTM.

All reported scores are produced by the executed notebooks; the best checkpoints are selected using validation combined score only.
