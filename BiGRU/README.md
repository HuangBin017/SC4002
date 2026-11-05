# NLP

Joint intent classification and slot filling with BiGRU, attention, and CRF models, including a GloVe variant.

- `model.ipynb` and `model_glove.ipynb`: notebooks.
- `joint_bigru_attention_crf.py` and `joint_bigru_attention_crf_glove.py`: model and training code.
- `1.0/`: dataset with its original license, notice, and citation.
- `stage2_outputs/` and `stage2_glove_outputs/`: saved model outputs.

Course slides and cache files are excluded from the repository. The large `glove.6B.100d.txt` embedding file is also excluded; download and extract the GloVe 6B 100-dimensional vectors separately and pass their location with `--glove`. See the training script's `--help` for options.
