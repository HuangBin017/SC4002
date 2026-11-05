
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
# 6. JOINT BiGRU + ATTENTION + CRF MODEL
# ============================================================

class JointBiGRUAttentionCRF(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_intents,
        num_tags,
        embed_dim=100,
        hidden_size=128,
        layers=2,
        dropout=0.3
    ):
        super().__init__()

        # Word embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # Shared BiGRU encoder
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0
        )

        encoder_dim = hidden_size * 2

        # Attention for intent classification
        self.attention = nn.Linear(
            encoder_dim,
            1
        )

        # Intent classification head
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_intents)
        )

        # Token-level slot emissions
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_tags)
        )

        # CRF sequence decoder
        self.crf = CRF(
            num_tags,
            batch_first=True
        )

    def forward(
        self,
        tokens,
        mask,
        tags=None,
        intents=None,
        slot_weight=1.0
    ):
        lengths = mask.long().sum(dim=1).cpu()

        # Embedding layer
        x = self.embedding_dropout(
            self.embedding(tokens)
        )

        # Pack sequences for the GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.encoder(packed)

        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=tokens.shape[1]
        )

        # Attention scores
        attention_logits = self.attention(
            encoded
        ).squeeze(-1)

        attention_logits = attention_logits.masked_fill(
            ~mask,
            -1e9
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1
        )

        # Sentence representation
        sentence_representation = (
            encoded * attention_weights.unsqueeze(-1)
        ).sum(dim=1)

        # Intent predictions
        intent_logits = self.intent_head(
            sentence_representation
        )

        # Slot predictions
        emissions = self.slot_head(encoded)

        result = {
            "intent_logits": intent_logits,
            "emissions": emissions
        }

        # Joint training loss
        if tags is not None and intents is not None:

            intent_loss = nn.functional.cross_entropy(
                intent_logits,
                intents
            )

            slot_loss = -self.crf(
                emissions,
                tags,
                mask=mask,
                reduction="mean"
            )

            total_loss = (
                intent_loss + slot_weight * slot_loss
            )

            result["loss"] = total_loss
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
def evaluate(model, loader, device, id_to_tag):
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
            intents
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
    """
    Initialize matching vocabulary words with GloVe vectors.
    """
    found = 0

    with torch.no_grad(), open(
        path,
        encoding="utf-8"
    ) as f:
        for line in f:
            fields = line.rstrip().split(" ")

            word = fields[0]

            if (
                word in vocab
                and len(fields) == embedding.embedding_dim + 1
            ):
                vector = torch.tensor(
                    [float(x) for x in fields[1:]]
                )

                embedding.weight[vocab[word]] = vector
                found += 1

    print(
        f"Loaded GloVe vectors for "
        f"{found}/{len(vocab)} tokens"
    )


# ============================================================
# 10. TRAINING PIPELINE
# ============================================================

def train(args):

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    splits = read_data(args.data)

    vocab, intent_to_id, tag_to_id = build_mappings(
        splits["train"]
    )

    id_to_tag = {
        idx: tag
        for tag, idx in tag_to_id.items()
    }

    datasets = {
        split: MassiveDataset(
            records,
            vocab,
            intent_to_id,
            tag_to_id
        )
        for split, records in splits.items()
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate
        )
        for split, dataset in datasets.items()
    }

    model = JointBiGRUAttentionCRF(
        vocab_size=len(vocab),
        num_intents=len(intent_to_id),
        num_tags=len(tag_to_id),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout
    )

    if args.glove:
        load_glove(
            args.glove,
            model.embedding,
            vocab
        )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    assert total_params <= 20_000_000

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        output_dir / "best_stage2_bigru_attention_crf.pt"
    )

    history = []

    best_score = -1
    best_epoch = -1

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

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                tokens,
                mask,
                tags,
                intents,
                args.slot_weight
            )

            loss = outputs["loss"]

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            batch_size = tokens.size(0)

            train_loss += loss.item() * batch_size
            seen += batch_size

        # Evaluate on validation data
        metrics = evaluate(
            model,
            loaders["dev"],
            device,
            id_to_tag
        )

        elapsed = time.perf_counter() - start_time

        score = (
            metrics["intent_accuracy"]
            + metrics["slot_micro_f1"]
        ) / 2

        row = {
            "epoch": epoch,
            "train_loss": train_loss / seen,
            "validation_loss": metrics["loss"],
            "validation_intent_accuracy":
                metrics["intent_accuracy"],
            "validation_slot_micro_f1":
                metrics["slot_micro_f1"],
            "selection_score": score,
            "epoch_seconds": elapsed
        }

        history.append(row)

        print(json.dumps(row, indent=2))

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save({
                "model": model.state_dict(),
                "vocab": vocab,
                "intent_to_id": intent_to_id,
                "tag_to_id": tag_to_id,
                "config": vars(args),
                "epoch": epoch,
                "validation_score": score,
                "total_parameters": total_params
            }, checkpoint_path)

            print("Saved new best checkpoint!")

    # Save history
    with open(
        output_dir / "training_history.json",
        "w"
    ) as f:
        json.dump(history, f, indent=2)

    # Plot loss curves
    fig, ax = plt.subplots()

    ax.plot(
        [h["epoch"] for h in history],
        [h["train_loss"] for h in history],
        label="Train"
    )

    ax.plot(
        [h["epoch"] for h in history],
        [h["validation_loss"] for h in history],
        label="Validation"
    )

    ax.set(
        xlabel="Epoch",
        ylabel="Loss",
        title="BiGRU + Attention + CRF Training"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "learning_curves.png",
        dpi=150
    )

    plt.close(fig)

    print("Best epoch:", best_epoch)
    print("Best validation score:", best_score)

    # Reload best checkpoint
    saved = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(saved["model"])

    for split in ["dev", "test"]:
        metrics = evaluate(
            model,
            loaders[split],
            device,
            id_to_tag
        )

        print(
            f"FINAL {split}: "
            f"Intent Accuracy = "
            f"{metrics['intent_accuracy']:.2f}% | "
            f"Slot Micro-F1 = "
            f"{metrics['slot_micro_f1']:.2f}%"
        )


# ============================================================
# 11. COMMAND LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--glove",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--output",
        default="stage2_outputs"
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=100
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--slot-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
