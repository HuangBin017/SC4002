# Stage-2 improvements (Arushi)

AI-assisted Stage-2 system built on adityamd's Stage-1 BiGRU + attention + CRF baseline
(`../BiGRU/model.ipynb`). It reuses that baseline's data loading, BIO conversion and span decoding
(`joint_bigru_attention_crf.py`), so both stages are scored by identical rules.

## Results (MASSIVE 1.0 en-US; every number is produced by the executed notebooks)

| Model version | Split | Total # params | Trainable params | Intent acc (%) | Slot micro-F1 (%) | Epochs |
|---|---|---|---|---|---|---|
| Stage-1 (`../BiGRU/model.ipynb`) | Validation | 1,044,435 | 1,044,435 | 84.01 | 78.98 | 30 |
| Stage-1 | Test | same | same | 82.82 | 75.63 | same |
| **Stage-2 (`model_stage2.ipynb`)** | Validation | 19,758,612 | 12,214,812 | **87.75** | **83.54** | 4 × best of 40 (29/34/37/31) |
| **Stage-2** | Test | same | same | **87.73** | **81.55** | same |

## Files

| Path | What it is |
|---|---|
| `model_stage2.ipynb` | **Final Stage-2 system.** GloVe 6B-300d plus a 4-seed ensemble, trained on a MacBook CPU (deterministic). Outputs are in `stage2_final_outputs/`. |
| `gpu_run/model_stage2_gpu.ipynb` | The same system trained on an NSCC ASPIRE2A A100 (`gpu_run/stage2.pbs`). Validation was lower (87.26 / 83.34), so it is **not** the selected run. Test: 87.49 / 81.38. |
| `model_stage2_noglove.ipynb` | Intermediate step: one Stage-2 model with no GloVe and no ensemble. Test 84.63 / 78.37. |
| `joint_bigru_stage2.py` | All Stage-2 model and training code. |
| `stage2_enhanced_outputs/` | Checkpoint, history and curves for the no-GloVe model. The final notebook's ablation table uses them. |
| `glove/` | GloVe 6B vectors (not in Git, ~1.9 GB). Download them from https://nlp.stanford.edu/projects/glove/ (mirror: https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip). |
| `joint_bigru_attention_crf.py`, `1.0/`, `stage2_outputs/` | **Links** to adityamd's files in `../BiGRU/`: the baseline module, the dataset and the Stage-1 checkpoint. |

## What Stage 2 changes vs. the Stage-1 baseline

1. **Frozen GloVe 300d + extended vocabulary.** The vocabulary is the training words plus the 20k most frequent
   GloVe words (picked by GloVe's own frequency ranking, never from dev/test). About 68% of test words that are
   unseen in training now get a real vector instead of `<unk>`.
2. **Character CNN.** Spelling features help with misspellings and rare names.
3. **Frequency-based word dropout.** During training, rare words are replaced by `<unk>` more often, so the
   model learns to cope with unknown words.
4. **BIO-constrained CRF.** Impossible tag sequences such as `O → I-time` are blocked.
5. **Intent-conditioned slot tagging.** The sentence vector is fed into every word's slot decision.
6. **Attention + max pooling** for intent, a wider BiGRU, AdamW with warm-up and cosine decay, and label smoothing.
7. **Ensemble of 4 seeds.** Members average their intent probabilities and CRF scores. All members share one frozen
   GloVe table, so the total stays under the 20M cap.

The notebook's development-history table also lists what was tried and rejected: a self-attention layer
(about −3.5 points on validation) and heavier dropout. All selection used the validation split only.

## Running
`jupyter nbconvert --to notebook --execute model_stage2.ipynb` from this folder. Needs `torch`, `pytorch-crf`,
`matplotlib`, `pandas` and `glove/glove.6B.300d.txt`. It takes ~80 min on an M2 CPU, or ~30 min on an A100
(set `"device": "cuda"` in the notebook's member config).
