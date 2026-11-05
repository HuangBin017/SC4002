"""Improved GloVe **300-dimensional** BiGRU + Attention + CRF for MASSIVE en-US.

Differences from the original GloVe experiment:
- lower LR for pretrained embeddings vs GRU/heads
- word dropout, stronger embedding/recurrent/output dropout, LayerNorm
- attention + masked max pooling for intent classification
- per-token CRF loss, label-smoothed intent loss
- validation-driven LR scheduler, early stopping, and best checkpoint
- independent output directory, incremental history and comparison plots

Requires: torch, pytorch-crf, numpy, matplotlib.
Example:
    python joint_bigru_attention_crf_glove_300d.py \
      --data /path/to/en-US.jsonl \
      --glove /path/to/glove.6B.300d.txt
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF


# ============================================================
# 1. CONSTANTS AND TOKENIZATION
# ============================================================

SLOT_RE = re.compile(r"\[([^\]:]+?)\s*:\s*([^\]]*?)\]")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

PAD = "<pad>"
UNK = "<unk>"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def annotated_slots(annotated):
    """
    Convert an annotated utterance into plain text and slot spans.

    Returns:
        plain_text
        [(slot_type, start_char, end_char), ...]
    """
    out = []
    slots = []
    cursor = 0
    length = 0

    for match in SLOT_RE.finditer(annotated):
        before = annotated[cursor:match.start()]
        out.append(before)
        length += len(before)

        value = match.group(2)
        start = length

        out.append(value)
        length += len(value)

        slots.append((match.group(1).strip(), start, length))
        cursor = match.end()

    out.append(annotated[cursor:])

    return "".join(out), slots


def tokenize_with_offsets(text):
    return [
        (m.group(), m.start(), m.end())
        for m in TOKEN_RE.finditer(text)
    ]


def record_to_tokens_tags(record):
    """
    Convert a MASSIVE record into tokens and BIO labels.
    """
    utt = record["utt"]
    clean, spans = annotated_slots(record["annot_utt"])

    if clean != utt:
        raise ValueError(
            "Annotated utterance does not reconstruct input exactly. "
            f"utt={utt!r}, reconstructed={clean!r}"
        )

    token_data = tokenize_with_offsets(utt)

    if not token_data:
        raise ValueError(f"No tokens in {utt!r}")

    tags = ["O"] * len(token_data)

    for typ, start, end in spans:
        covered = [
            i
            for i, (_, a, b) in enumerate(token_data)
            if a >= start and b <= end
        ]

        if (
            not covered
            or token_data[covered[0]][1] != start
            or token_data[covered[-1]][2] != end
        ):
            raise ValueError(
                f"Slot boundary is not token aligned: "
                f"{utt!r}, {(typ, start, end)}"
            )

        for j, idx in enumerate(covered):
            if tags[idx] != "O":
                raise ValueError(
                    "Overlapping slot labels are unsupported"
                )

            tags[idx] = (
                "B-" if j == 0 else "I-"
            ) + typ

    return [tok for tok, _, _ in token_data], tags


# ============================================================
# 2. LOAD MASSIVE DATASET
# ============================================================

def read_data(path):
    """
    Read MASSIVE en-US.jsonl using its official partitions.
    """
    splits = {
        "train": [],
        "dev": [],
        "test": []
    }

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                partition = record["partition"]

                if partition in splits:
                    splits[partition].append(record)

    expected = {
        "train": 11514,
        "dev": 2033,
        "test": 2974
    }

    actual = {
        key: len(value)
        for key, value in splits.items()
    }

    print("Dataset split sizes:", actual)

    assert actual == expected, (
        "Dataset does not match official MASSIVE 1.0 "
        "English splits."
    )

    return splits


# ============================================================
# 3. BUILD VOCABULARY AND LABEL MAPPINGS
# ============================================================

def build_mappings(train):
    """
    Build all mappings using training examples only.
    """
    token_counts = Counter()
    intents = set()
    labels = {"O"}

    for record in train:
        words, tags = record_to_tokens_tags(record)

        token_counts.update(
            word.lower() for word in words
        )

        intents.add(record["intent"])
        labels.update(tags)

    # A slot may be one token in training but span multiple tokens in
    # validation/test. Reserve both BIO forms for every training slot type.
    slot_types = {tag.split("-", 1)[1] for tag in labels if tag != "O"}
    for slot_type in slot_types:
        labels.update({f"B-{slot_type}", f"I-{slot_type}"})

    vocab = {
        PAD: 0,
        UNK: 1
    }

    for word, _ in token_counts.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)

    intent_to_id = {
        label: idx
        for idx, label in enumerate(sorted(intents))
    }

    tag_to_id = {
        label: idx
        for idx, label in enumerate(
            ["O"] + sorted(labels - {"O"})
        )
    }

    print("Vocabulary size:", len(vocab))
    print("Number of intents:", len(intent_to_id))
    print("Number of BIO tags:", len(tag_to_id))

    assert len(intent_to_id) == 60

    return vocab, intent_to_id, tag_to_id


# ============================================================
# 4. PYTORCH DATASET
# ============================================================

class MassiveDataset(Dataset):

    def __init__(
        self,
        records,
        vocab,
        intent_to_id,
        tag_to_id
    ):
        self.examples = []

        for record in records:
            words, tags = record_to_tokens_tags(record)

            token_ids = [
                vocab.get(word.lower(), 1)
                for word in words
            ]

            tag_ids = [
                tag_to_id[tag]
                for tag in tags
            ]

            _, slot_spans = annotated_slots(
                record["annot_utt"]
            )

            gold_slots = {
                (typ, record["utt"][start:end])
                for typ, start, end in slot_spans
            }

            self.examples.append({
                "tokens": torch.tensor(
                    token_ids,
                    dtype=torch.long
                ),
                "tags": torch.tensor(
                    tag_ids,
                    dtype=torch.long
                ),
                "intent": intent_to_id[
                    record["intent"]
                ],
                "utt": record["utt"],
                "gold_slots": gold_slots
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# 5. BATCH PADDING
# ============================================================

def collate(batch):
    """
    Pad variable-length token and tag sequences.
    """
    max_len = max(
        len(example["tokens"])
        for example in batch
    )

    batch_size = len(batch)

    tokens = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.long
    )

    tags = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.long
    )

    mask = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.bool
    )

    for i, example in enumerate(batch):
        n = len(example["tokens"])

        tokens[i, :n] = example["tokens"]
        tags[i, :n] = example["tags"]
        mask[i, :n] = True

    return {
        "tokens": tokens,
        "tags": tags,
        "mask": mask,
        "intents": torch.tensor(
            [example["intent"] for example in batch],
            dtype=torch.long
        ),
        "utts": [
            example["utt"]
            for example in batch
        ],
        "gold_slots": [
            example["gold_slots"]
            for example in batch
        ]
    }


# ============================================================
# 6. IMPROVED JOINT BiGRU + ATTENTION + CRF MODEL
# ============================================================

class JointBiGRUAttentionCRF(nn.Module):
    """Same shared BiGRU/CRF design, with regularized embeddings & intent pooling."""

    def __init__(self, vocab_size, num_intents, num_tags,
                 embed_dim=300, hidden_size=128, layers=2,
                 dropout=0.45, word_dropout=0.08,
                 intent_label_smoothing=0.05):
        super().__init__()
        self.word_dropout = word_dropout
        self.intent_label_smoothing = intent_label_smoothing
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder = nn.GRU(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=layers, batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        encoder_dim = hidden_size * 2
        self.encoder_norm = nn.LayerNorm(encoder_dim)
        self.encoder_dropout = nn.Dropout(dropout)

        # Attention pools contextual token embeddings; max pooling captures
        # discriminative keywords that attention may otherwise underweight.
        self.attention = nn.Linear(encoder_dim, 1)
        self.intent_head = nn.Sequential(
            nn.LayerNorm(encoder_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 2, num_intents),
        )

        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_tags),
        )
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, tokens, mask, tags=None, intents=None, slot_weight=1.5):
        # The UNK id is 1; never replace PAD tokens (id 0).
        # Word dropout forces the encoder to work with unseen/OOV words.
        if self.training and self.word_dropout > 0:
            eligible = (tokens > 1) & mask
            replace = (torch.rand(tokens.shape, device=tokens.device)
                       < self.word_dropout) & eligible
            tokens = tokens.masked_fill(replace, 1)

        lengths = mask.long().sum(dim=1).cpu()
        x = self.embedding_dropout(self.embedding(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=tokens.shape[1]
        )
        encoded = self.encoder_dropout(self.encoder_norm(encoded))

        attention_logits = self.attention(encoded).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_context = (encoded * attention_weights.unsqueeze(-1)).sum(dim=1)

        max_context = encoded.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        sentence_representation = torch.cat([attention_context, max_context], dim=-1)
        intent_logits = self.intent_head(sentence_representation)
        emissions = self.slot_head(encoded)

        result = {"intent_logits": intent_logits, "emissions": emissions}
        if tags is not None and intents is not None:
            intent_loss = nn.functional.cross_entropy(
                intent_logits, intents, label_smoothing=self.intent_label_smoothing
            )
            # Normalized across NON-PAD tokens, rather than averaging the
            # summed sequence losses. This makes it less sensitive to utterance length.
            slot_loss = -self.crf(emissions, tags, mask=mask, reduction="token_mean")
            result["loss"] = intent_loss + slot_weight * slot_loss
            result["intent_loss"] = intent_loss.detach().item()
            result["slot_loss"] = slot_loss.detach().item()
        return result


# ============================================================
# 7. DECODE BIO TAGS INTO EXACT SLOT SPANS
# ============================================================

def decode_slots(utt, tag_ids, id_to_tag):
    """
    Decode BIO predictions into (slot_type, slot_text) pairs.
    """
    token_data = tokenize_with_offsets(utt)

    spans = set()

    current_type = None
    start = None
    end = None

    def close_slot():
        if current_type is not None:
            spans.add(
                (current_type, utt[start:end])
            )

    for (_, a, b), tag_id in zip(token_data, tag_ids):
        label = id_to_tag[tag_id]

        if label == "O":
            close_slot()
            current_type = None
            start = None
            end = None
            continue

        prefix, typ = label.split("-", 1)

        if prefix == "B" or current_type != typ:
            close_slot()
            current_type = typ
            start = a

        end = b

    close_slot()

    return spans


# ============================================================
# 8. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, id_to_tag, slot_weight=1.5):
    model.eval()

    correct = 0
    total = 0

    tp = 0
    fp = 0
    fn = 0

    loss_sum = 0.0

    for batch in loader:
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        tags = batch["tags"].to(device)
        intents = batch["intents"].to(device)

        outputs = model(
            tokens,
            mask,
            tags,
            intents,
            slot_weight,
        )

        batch_size = len(batch["utts"])

        loss_sum += (
            outputs["loss"].item() * batch_size
        )

        intent_predictions = outputs[
            "intent_logits"
        ].argmax(dim=1)

        correct += (
            intent_predictions == intents
        ).sum().item()

        total += batch_size

        slot_predictions = model.crf.decode(
            outputs["emissions"],
            mask=mask
        )

        for utt, predicted_tags, gold in zip(
            batch["utts"],
            slot_predictions,
            batch["gold_slots"]
        ):
            predicted = decode_slots(
                utt,
                predicted_tags,
                id_to_tag
            )

            tp += len(predicted & gold)
            fp += len(predicted - gold)
            fn += len(gold - predicted)

    intent_accuracy = 100 * correct / total

    slot_micro_f1 = (
        100 * 2 * tp / max(1, 2 * tp + fp + fn)
    )

    return {
        "loss": loss_sum / total,
        "intent_accuracy": intent_accuracy,
        "slot_micro_f1": slot_micro_f1
    }


# ============================================================
# 9. OPTIONAL GLOVE EMBEDDINGS
# ============================================================

def load_glove(path, embedding, vocab):
    """Initialize in-vocabulary embeddings from a GloVe 300d .txt file.

    Only words in the training vocabulary are loaded. Unmatched words retain
    their randomly initialized vectors. The PAD embedding stays all zeros.
    Raises an error if dimensions mismatch or no vocabulary words match.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"GloVe file not found: {path}. Expected a text file such as "
            "glove.6B.300d.txt (not the downloaded ZIP archive)."
        )

    found = set()
    expected_dim = embedding.embedding_dim
    with torch.no_grad(), path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            word = fields[0]
            if word not in vocab or word in {PAD, UNK}:
                continue
            if len(fields) != expected_dim + 1:
                raise ValueError(
                    f"GloVe vector dimension mismatch at line {line_no} for "
                    f"{word!r}: expected {expected_dim}, got {len(fields)-1}. "
                    "Use a GloVe file that matches --embed-dim."
                )
            try:
                vector = np.asarray(fields[1:], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric GloVe vector on line {line_no}: {word!r}"
                ) from exc
            embedding.weight[vocab[word]].copy_(torch.from_numpy(vector))
            found.add(word)

        embedding.weight[vocab[PAD]].zero_()

    if not found:
        raise ValueError(
            f"No matching GloVe words found in {path}; check the file and "
            "the --embed-dim argument. Training has NOT started."
        )

    word_count = max(1, len(vocab) - 2)
    coverage = {
        "glove_file": str(path.resolve()),
        "embedding_dimension": expected_dim,
        "matched_tokens": len(found),
        "vocabulary_tokens_excluding_special": word_count,
        "coverage_percent": round(100 * len(found) / word_count, 2),
        "embedding_fine_tuned": True,
    }
    print(
        f"Initialized {len(found):,}/{word_count:,} vocabulary words "
        f"from GloVe ({coverage['coverage_percent']:.2f}% coverage)."
    )
    return coverage


# ============================================================
# 10. TRAINING PIPELINE
# ============================================================

def _make_optimizer(model, args):
    """Keep pretrained GloVe vectors on a smaller learning rate."""
    embedding_params = list(model.embedding.parameters())
    other_params = [p for name, p in model.named_parameters()
                    if not name.startswith("embedding.") and p.requires_grad]
    groups = [{"params": other_params, "lr": args.lr}]
    if not args.freeze_embeddings:
        groups.append({"params": embedding_params, "lr": args.embedding_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _plot_history(history, output_dir):
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [x["train_loss"] for x in history], label="Training")
    ax.plot(epochs, [x["validation_loss"] for x in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Joint loss (rebalanced)",
           title="Improved GloVe 300d BiGRU: loss curves")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [x["validation_intent_accuracy"] for x in history],
            label="Intent accuracy (%)")
    ax.plot(epochs, [x["validation_slot_micro_f1"] for x in history],
            label="Slot micro-F1 (%)")
    ax.plot(epochs, [x["selection_score"] for x in history],
            label="Combined validation score")
    ax.set(xlabel="Epoch", ylabel="Validation metric (%)",
           title="Improved GloVe 300d BiGRU: validation metrics")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_metrics.png", dpi=150)
    plt.close(fig)


def train(args):
    if args.embed_dim != 300:
        raise ValueError(
            f"This experiment requires 300-dimensional GloVe embeddings; "
            f"received --embed-dim={args.embed_dim}."
        )
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device, flush=True)
    splits = read_data(args.data)
    vocab, intent_to_id, tag_to_id = build_mappings(splits["train"])
    id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}

    datasets = {
        split: MassiveDataset(records, vocab, intent_to_id, tag_to_id)
        for split, records in splits.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(split == "train"),
            collate_fn=collate,
            generator=generator if split == "train" else None,
        ) for split, dataset in datasets.items()
    }

    model = JointBiGRUAttentionCRF(
        vocab_size=len(vocab), num_intents=len(intent_to_id),
        num_tags=len(tag_to_id), embed_dim=args.embed_dim,
        hidden_size=args.hidden_size, layers=args.layers,
        dropout=args.dropout, word_dropout=args.word_dropout,
        intent_label_smoothing=args.intent_label_smoothing,
    )
    print(f"Model embedding dimension: {model.embedding.embedding_dim} (GloVe 300d)")
    glove_stats = load_glove(args.glove, model.embedding, vocab)
    if args.freeze_embeddings:
        model.embedding.weight.requires_grad_(False)
        glove_stats["embedding_fine_tuned"] = False
    else:
        print(f"Fine-tuning GloVe with LR={args.embedding_lr:g}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    assert total_params <= 20_000_000, "Model exceeds the 20M-parameter limit"
    model = model.to(device)

    optimizer = _make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_decay,
        patience=args.lr_patience, min_lr=args.min_lr,
    )
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_bigru_attention_crf_glove_300d.pt"
    (output_dir / "glove_coverage.json").write_text(
        json.dumps(glove_stats, indent=2), encoding="utf-8"
    )

    history = []
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        model.train()
        train_loss = 0.0
        seen = 0

        for batch in loaders["train"]:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            tags = batch["tags"].to(device)
            intents = batch["intents"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(tokens, mask, tags, intents, args.slot_weight)
            loss = outputs["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = tokens.size(0)
            train_loss += loss.item() * batch_size
            seen += batch_size

        metrics = evaluate(model, loaders["dev"], device,
                           id_to_tag, slot_weight=args.slot_weight)
        score = (metrics["intent_accuracy"] + metrics["slot_micro_f1"]) / 2
        scheduler.step(score)
        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "vocab": vocab, "intent_to_id": intent_to_id,
                "tag_to_id": tag_to_id, "epoch": epoch,
                "validation_score": score,
                "total_parameters": total_params,
                "glove_stats": glove_stats,
                "config": vars(args),
            }, checkpoint_path)
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss / seen,
            "validation_loss": metrics["loss"],
            "validation_intent_accuracy": metrics["intent_accuracy"],
            "validation_slot_micro_f1": metrics["slot_micro_f1"],
            "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "embedding_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            "epoch_seconds": time.perf_counter() - start_time,
        }
        history.append(row)
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        print(f"\n{'-'*58}", flush=True)
        print(f"EPOCH {epoch} RESULTS", flush=True)
        print(f"{'-'*58}", flush=True)
        print(f"Training Loss:       {row['train_loss']:.4f}")
        print(f"Validation Loss:     {metrics['loss']:.4f}")
        print(f"Intent Accuracy:     {metrics['intent_accuracy']:.2f}%")
        print(f"Slot Micro-F1:       {metrics['slot_micro_f1']:.2f}%")
        print(f"Combined Score:      {score:.2f}")
        print(f"LR (encoder):        {row['learning_rate']:.6g}")
        print(f"Epoch Time:          {row['epoch_seconds']:.2f} seconds")
        print(f"Best: epoch {best_epoch} ({best_score:.2f}) | "
              f"{'NEW CHECKPOINT' if improved else f'no improvement {stale_epochs}/{args.early_stop_patience}'}",
              flush=True)

        if stale_epochs >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; restoring epoch {best_epoch}.")
            break

    _plot_history(history, output_dir)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    final_metrics = {}
    for split in ("dev", "test"):
        final_metrics[split] = evaluate(model, loaders[split], device,
                                        id_to_tag, slot_weight=args.slot_weight)
        m = final_metrics[split]
        print(f"FINAL {split.upper()}: intent accuracy={m['intent_accuracy']:.2f}% | "
              f"slot micro-F1={m['slot_micro_f1']:.2f}%")

    summary = {
        "best_epoch": best_epoch, "best_validation_score": best_score,
        "epochs_run": len(history), "glove": glove_stats,
        "metrics": final_metrics,
        "checkpoint": str(checkpoint_path),
        "note": "Joint loss is on a different scale from the baseline because CRF uses token_mean.",
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ============================================================
# 11. CLI
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Regularized joint BiGRU + Attention + CRF with pretrained GloVe 300d"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("stage2_glove_300d_outputs"))
    parser.add_argument("--embed-dim", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--word-dropout", type=float, default=0.08)
    parser.add_argument("--intent-label-smoothing", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embedding-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--slot-weight", type=float, default=1.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-embeddings", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
