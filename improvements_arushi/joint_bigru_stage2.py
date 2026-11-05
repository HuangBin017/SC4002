"""Stage-2 (AI-enhanced) joint BiGRU model for MASSIVE 1.0 en-US.

Builds on the Stage-1 BiGRU + attention + CRF baseline
(joint_bigru_attention_crf.py) and reuses its data loading, BIO conversion
and span decoding so that both stages are scored with identical rules.

Changes relative to the Stage-1 baseline (see README / report for the why):
  1. Character-CNN word features            -> handles OOV / misspelt words
  2. Frequency-based word dropout to <unk>  -> the <unk> vector actually gets
                                               trained (it never appears in
                                               the baseline's training data)
  3. Wider BiGRU (256 per direction); an optional residual self-attention
     block is implemented but disabled because it lowered the dev score
  4. Intent pooling = attention pooling + max pooling
  5. Intent-conditioned slot emissions (sentence vector fed to slot head)
  6. CRF with hard BIO constraints (no O -> I-x, no B-x -> I-y)
  7. Label smoothing on the intent loss
  8. AdamW + linear warm-up + cosine decay, trained on CPU (deterministic and
     faster than MPS for this model size)

No pretrained model of any kind is used: every weight, including the word
embeddings, is randomly initialised and trained on the MASSIVE train split.

Example:
    python joint_bigru_stage2.py --data 1.0/data/en-US.jsonl
"""

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from joint_bigru_attention_crf import (
    PAD,
    UNK,
    annotated_slots,
    build_mappings,
    decode_slots,
    read_data,
    record_to_tokens_tags,
    seed_everything,
)

MAX_WORD_LEN = 20
SINGLE_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


# ============================================================
# 1. SHARED METRICS (identical scoring for every experiment)
# ============================================================

def compute_metrics(pred_intents, gold_intents, pred_slots, gold_slots):
    """Intent accuracy and exact-match [slot type: slot span] micro F1.

    pred_slots / gold_slots: one set of (slot_type, span_text) per utterance.
    A true positive needs both the type and the exact span text to match.
    """
    assert len(pred_intents) == len(gold_intents) == len(pred_slots) == len(gold_slots)
    correct = sum(int(p == g) for p, g in zip(pred_intents, gold_intents))
    tp = fp = fn = 0
    for pred, gold in zip(pred_slots, gold_slots):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    intent_accuracy = 100 * correct / len(gold_intents)
    slot_micro_f1 = 100 * 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "intent_accuracy": intent_accuracy,
        "slot_micro_f1": slot_micro_f1,
        "combined": (intent_accuracy + slot_micro_f1) / 2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 2. PREPROCESSING: WORDS + CHARACTERS + BIO TAGS
# ============================================================

def build_char_vocab(train):
    """Character inventory from the training split only (0 = pad, 1 = unk)."""
    chars = Counter()
    for record in train:
        words, _ = record_to_tokens_tags(record)
        for word in words:
            chars.update(word.lower())
    char_to_id = {"<cpad>": 0, "<cunk>": 1}
    for ch, _ in chars.most_common():
        char_to_id[ch] = len(char_to_id)
    return char_to_id


def word_frequencies(train, vocab):
    counts = torch.zeros(len(vocab))
    for record in train:
        words, _ = record_to_tokens_tags(record)
        for word in words:
            counts[vocab[word.lower()]] += 1
    return counts


class Stage2Dataset(Dataset):
    def __init__(self, records, vocab, char_to_id, intent_to_id, tag_to_id):
        self.examples = []
        for record in records:
            words, tags = record_to_tokens_tags(record)
            chars = [
                [char_to_id.get(c, 1) for c in word.lower()[:MAX_WORD_LEN]]
                for word in words
            ]
            _, spans = annotated_slots(record["annot_utt"])
            self.examples.append({
                "tokens": torch.tensor([vocab.get(w.lower(), 1) for w in words]),
                "chars": chars,
                "tags": torch.tensor([tag_to_id[t] for t in tags]),
                "intent": intent_to_id[record["intent"]],
                "utt": record["utt"],
                "gold_slots": {(typ, record["utt"][a:b]) for typ, a, b in spans},
                "record": record,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(batch):
    max_len = max(len(ex["tokens"]) for ex in batch)
    max_word = max(len(w) for ex in batch for w in ex["chars"])
    size = len(batch)
    tokens = torch.zeros(size, max_len, dtype=torch.long)
    tags = torch.zeros(size, max_len, dtype=torch.long)
    mask = torch.zeros(size, max_len, dtype=torch.bool)
    chars = torch.zeros(size, max_len, max_word, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex["tokens"])
        tokens[i, :n] = ex["tokens"]
        tags[i, :n] = ex["tags"]
        mask[i, :n] = True
        for j, word in enumerate(ex["chars"]):
            chars[i, j, :len(word)] = torch.tensor(word)
    return {
        "tokens": tokens,
        "chars": chars,
        "tags": tags,
        "mask": mask,
        "intents": torch.tensor([ex["intent"] for ex in batch]),
        "utts": [ex["utt"] for ex in batch],
        "gold_slots": [ex["gold_slots"] for ex in batch],
    }


# ============================================================
# 3. CRF WITH HARD BIO CONSTRAINTS
# ============================================================

class ConstrainedCRF(nn.Module):
    """Linear-chain CRF whose transition matrix forbids invalid BIO moves.

    Forbidden: START -> I-x, O -> I-x, B-x/I-x -> I-y for y != x.
    """

    NEG = -1e4

    def __init__(self, id_to_tag):
        super().__init__()
        n = len(id_to_tag)
        self.num_tags = n
        self.transitions = nn.Parameter(torch.empty(n, n).uniform_(-0.1, 0.1))
        self.start_transitions = nn.Parameter(torch.empty(n).uniform_(-0.1, 0.1))
        self.end_transitions = nn.Parameter(torch.empty(n).uniform_(-0.1, 0.1))

        labels = [id_to_tag[i] for i in range(n)]
        trans_mask = torch.zeros(n, n)
        start_mask = torch.zeros(n)
        for j, to_tag in enumerate(labels):
            if not to_tag.startswith("I-"):
                continue
            typ = to_tag[2:]
            start_mask[j] = self.NEG
            for i, from_tag in enumerate(labels):
                if from_tag not in (f"B-{typ}", f"I-{typ}"):
                    trans_mask[i, j] = self.NEG
        self.register_buffer("trans_mask", trans_mask)
        self.register_buffer("start_mask", start_mask)

    def _scores(self):
        return (
            self.transitions + self.trans_mask,
            self.start_transitions + self.start_mask,
            self.end_transitions,
        )

    def neg_log_likelihood(self, emissions, tags, mask):
        """Mean negative log-likelihood over the batch."""
        trans, start, end = self._scores()
        batch, length, _ = emissions.shape
        maskf = mask.float()
        idx = torch.arange(batch, device=emissions.device)

        # Score of the gold path.
        score = start[tags[:, 0]] + emissions[idx, 0, tags[:, 0]]
        for t in range(1, length):
            step = trans[tags[:, t - 1], tags[:, t]] + emissions[idx, t, tags[:, t]]
            score = score + step * maskf[:, t]
        last = tags.gather(1, (mask.long().sum(1) - 1).unsqueeze(1)).squeeze(1)
        score = score + end[last]

        # Log partition function (forward algorithm).
        alpha = start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, length):
            nxt = torch.logsumexp(
                alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1),
                dim=1,
            )
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        log_z = torch.logsumexp(alpha + end.unsqueeze(0), dim=1)
        return (log_z - score).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi decoding; returns a list of tag-id lists (unpadded)."""
        trans, start, end = self._scores()
        batch, length, _ = emissions.shape
        score = start.unsqueeze(0) + emissions[:, 0]
        history = []
        for t in range(1, length):
            cand = score.unsqueeze(2) + trans.unsqueeze(0)
            best, best_idx = cand.max(dim=1)
            nxt = best + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), nxt, score)
            history.append(best_idx)
        score = score + end.unsqueeze(0)
        lengths = mask.long().sum(1).tolist()
        last_best = score.argmax(1).tolist()
        history = [h.cpu() for h in history]
        paths = []
        for b in range(batch):
            tag = last_best[b]
            path = [tag]
            for t in range(lengths[b] - 1, 0, -1):
                tag = history[t - 1][b, tag].item()
                path.append(tag)
            paths.append(path[::-1])
        return paths


# ============================================================
# 4. MODEL
# ============================================================

class CharCNN(nn.Module):
    """Character CNN with max-over-time pooling.

    Each convolution is written as unfold + Linear, which is mathematically a
    Conv1d but ~10x faster than nn.Conv1d on Apple-silicon CPUs. Only real
    (non-padding) tokens are encoded; padding positions get a zero vector.
    """

    def __init__(self, num_chars, char_dim, filters, kernels):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, char_dim, padding_idx=0)
        self.kernels = tuple(kernels)
        self.convs = nn.ModuleList(nn.Linear(char_dim * k, filters) for k in self.kernels)
        self.out_dim = filters * len(self.kernels)

    def forward(self, chars, mask):
        batch, length, _ = chars.shape
        words = chars[mask]                                  # (N, W)
        words = words[:, : int((words != 0).sum(1).max())]   # trim char padding
        x = self.embedding(words).transpose(1, 2)            # (N, C, W)
        char_mask = words != 0
        feats = []
        for k, proj in zip(self.kernels, self.convs):
            xp = nn.functional.pad(x, (k // 2, k // 2))
            windows = xp.unfold(2, k, 1)[:, :, : words.shape[1]]          # (N, C, W, k)
            windows = windows.permute(0, 2, 1, 3).reshape(*words.shape, -1)
            h = torch.relu(proj(windows)).masked_fill(~char_mask.unsqueeze(-1), 0.0)
            feats.append(h.max(dim=1).values)
        out = chars.new_zeros(batch, length, self.out_dim, dtype=feats[0].dtype)
        out[mask] = torch.cat(feats, dim=1)
        return out


class JointBiGRUStage2(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_chars,
        num_intents,
        id_to_tag,
        unk_prob,
        embed_dim=300,
        char_dim=48,
        char_filters=64,
        char_kernels=(2, 3, 4),
        hidden_size=256,
        layers=2,
        heads=8,
        use_self_attention=False,
        intent_context_dim=128,
        dropout=0.3,
        emb_dropout=0.15,
        label_smoothing=0.1,
        glove_vectors=None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.train_vocab_size = vocab_size
        # Not a parameter: per-word probability of being replaced by <unk>
        # during training (frequency-based word dropout).
        self.register_buffer("unk_prob", unk_prob.float())

        # Trainable embedding for words seen in the training split only.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Optional frozen GloVe table over the extended vocabulary
        # (training words + the most frequent GloVe words).
        self.glove = None
        glove_dim = 0
        if glove_vectors is not None:
            self.glove = nn.Embedding.from_pretrained(
                glove_vectors.float(), freeze=True, padding_idx=0
            )
            glove_dim = glove_vectors.shape[1]
        self.char_cnn = CharCNN(num_chars, char_dim, char_filters, char_kernels)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.encoder = nn.GRU(
            glove_dim + embed_dim + self.char_cnn.out_dim,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        enc_dim = 2 * hidden_size
        self.dropout = nn.Dropout(dropout)

        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                enc_dim, heads, dropout=dropout, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(enc_dim)

        # Intent: attention pooling + max pooling.
        self.attn_score = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2), nn.Tanh(), nn.Linear(enc_dim // 2, 1)
        )
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * enc_dim, enc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, num_intents),
        )

        # Slots: token states conditioned on the sentence (intent) vector.
        self.intent_context = nn.Linear(2 * enc_dim, intent_context_dim)
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim + intent_context_dim, enc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim // 2, len(id_to_tag)),
        )
        self.crf = ConstrainedCRF(id_to_tag)

    def forward(self, tokens, chars, mask, tags=None, intents=None):
        # Ids beyond the training vocabulary only exist in the GloVe table.
        train_ids = tokens.masked_fill(tokens >= self.train_vocab_size, 1)
        if self.training:
            drop = torch.rand(tokens.shape, device=tokens.device) < self.unk_prob[tokens]
            train_ids = train_ids.masked_fill(drop & mask, 1)  # 1 = <unk>

        parts = [self.embedding(train_ids), self.char_cnn(chars, mask)]
        if self.glove is not None:
            parts.insert(0, self.glove(tokens))
        x = torch.cat(parts, dim=-1)
        x = self.emb_dropout(x)

        lengths = mask.long().sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=tokens.shape[1]
        )
        enc = self.dropout(enc)

        if self.use_self_attention:
            attn, _ = self.self_attn(enc, enc, enc, key_padding_mask=~mask, need_weights=False)
            enc = self.attn_norm(enc + self.dropout(attn))

        scores = self.attn_score(enc).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pool = (enc * weights).sum(1)
        max_pool = enc.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        sentence = torch.cat([attn_pool, max_pool], dim=-1)

        intent_logits = self.intent_head(sentence)

        context = torch.tanh(self.intent_context(sentence))
        context = context.unsqueeze(1).expand(-1, enc.shape[1], -1)
        emissions = self.slot_head(torch.cat([enc, context], dim=-1))

        result = {"intent_logits": intent_logits, "emissions": emissions}
        if tags is not None and intents is not None:
            intent_loss = nn.functional.cross_entropy(
                intent_logits, intents, label_smoothing=self.label_smoothing
            )
            slot_loss = self.crf.neg_log_likelihood(emissions, tags, mask)
            result["loss"] = intent_loss + slot_loss
            result["intent_loss"] = intent_loss.item()
            result["slot_loss"] = slot_loss.item()
        return result


class AveragedCRF(nn.Module):
    """Decodes with the mean of several members' CRF scores."""

    def __init__(self, crfs):
        super().__init__()
        self.crfs = crfs

    def _scores(self):
        scores = [crf._scores() for crf in self.crfs]
        return tuple(torch.stack(parts).mean(0) for parts in zip(*scores))

    decode = ConstrainedCRF.decode


class JointEnsemble(nn.Module):
    """Ensemble of independently trained joint models (one system, two outputs).

    Intent: mean of member probabilities. Slots: mean of member emission and
    transition scores, then one Viterbi decode. The frozen GloVe table is
    shared by all members, so it is stored and counted only once.
    """

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        shared = members[0].glove
        for member in members[1:]:
            if shared is not None:
                assert torch.equal(member.glove.weight, shared.weight)
                member.glove = shared
        self.crf = AveragedCRF([m.crf for m in members])

    def forward(self, tokens, chars, mask, tags=None, intents=None):
        outs = [m(tokens, chars, mask, tags, intents) for m in self.members]
        probs = torch.stack([o["intent_logits"].softmax(-1) for o in outs]).mean(0)
        result = {
            "intent_logits": probs.log(),
            "emissions": torch.stack([o["emissions"] for o in outs]).mean(0),
        }
        if tags is not None and intents is not None:
            result["loss"] = torch.stack([o["loss"] for o in outs]).mean()
        return result


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 5. PREDICTION + EVALUATION
# ============================================================

@torch.no_grad()
def predict(model, loader, device, id_to_tag):
    """Returns loss, predictions and gold labels for every example."""
    model.eval()
    loss_sum, total = 0.0, 0
    out = {"pred_intents": [], "gold_intents": [], "pred_slots": [], "gold_slots": [], "utts": []}
    for batch in loader:
        tokens = batch["tokens"].to(device)
        chars = batch["chars"].to(device)
        mask = batch["mask"].to(device)
        tags = batch["tags"].to(device)
        intents = batch["intents"].to(device)
        result = model(tokens, chars, mask, tags, intents)
        size = tokens.shape[0]
        loss_sum += result["loss"].item() * size
        total += size
        out["pred_intents"] += result["intent_logits"].argmax(1).tolist()
        out["gold_intents"] += intents.tolist()
        for utt, path, gold in zip(
            batch["utts"], model.crf.decode(result["emissions"], mask), batch["gold_slots"]
        ):
            out["pred_slots"].append(decode_slots(utt, path, id_to_tag))
            out["gold_slots"].append(gold)
            out["utts"].append(utt)
    out["loss"] = loss_sum / total
    return out


def evaluate(model, loader, device, id_to_tag):
    pred = predict(model, loader, device, id_to_tag)
    metrics = compute_metrics(
        pred["pred_intents"], pred["gold_intents"], pred["pred_slots"], pred["gold_slots"]
    )
    metrics["loss"] = pred["loss"]
    return metrics


# ============================================================
# 6. TRAINING PIPELINE
# ============================================================

DEFAULT_CONFIG = {
    "seed": 42,
    "epochs": 40,
    "batch_size": 32,
    "lr": 2e-3,
    "weight_decay": 0.01,
    "warmup_epochs": 1,
    "clip": 5.0,
    "unk_z": 0.25,       # p(<unk>) = unk_z / (unk_z + train_count)
    "embed_dim": 300,
    "char_dim": 48,
    "char_filters": 64,
    "char_kernels": [2, 3, 4],
    "hidden_size": 256,
    "layers": 2,
    "heads": 8,
    "use_self_attention": False,  # tried; hurt validation score (see notebook)
    "intent_context_dim": 128,
    "dropout": 0.3,
    "emb_dropout": 0.15,
    "label_smoothing": 0.1,
    "threads": 8,
    "device": "cpu",     # "cpu" (deterministic) or "cuda"/"auto" for an NVIDIA GPU
}

MODEL_KEYS = [
    "embed_dim", "char_dim", "char_filters", "char_kernels", "hidden_size", "layers",
    "heads", "use_self_attention", "intent_context_dim", "dropout", "emb_dropout",
    "label_smoothing",
]


def load_glove_table(path, vocab, extra_words):
    """Frozen GloVe table for the training vocabulary plus extra GloVe words.

    The extra words are simply the `extra_words` most frequent GloVe entries
    (GloVe files are sorted by corpus frequency) that our tokenizer can
    produce and that are not already in the training vocabulary. The choice
    never looks at the validation or test split.
    Returns (extended vocab, vectors, info). Rows with no GloVe vector are 0.
    """
    vocab = dict(vocab)
    train_size = len(vocab)
    vectors, found, added = {}, 0, 0
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            word, rest = line.rstrip("\n").split(" ", 1)
            if word in (PAD, UNK):
                continue
            if word in vocab:
                found += 1
            elif added < extra_words and SINGLE_TOKEN_RE.fullmatch(word):
                vocab[word] = len(vocab)
                added += 1
            else:
                continue
            vectors[vocab[word]] = np.asarray(rest.split(" "), dtype=np.float32)
    dim = len(next(iter(vectors.values())))
    table = torch.zeros(len(vocab), dim)
    for idx, vec in vectors.items():
        table[idx] = torch.from_numpy(vec)
    info = {
        "glove_file": Path(path).name,
        "dimension": dim,
        "train_vocab_words_with_vector": found,
        "train_vocab_size_excluding_special": train_size - 2,
        "train_vocab_coverage_percent": round(100 * found / (train_size - 2), 2),
        "extra_glove_words_added": added,
        "extended_vocab_size": len(vocab),
    }
    return vocab, table, info


def prepare(data_path, glove_path=None, glove_extra_words=0):
    splits = read_data(data_path)
    vocab, intent_to_id, tag_to_id = build_mappings(splits["train"])
    train_vocab_size = len(vocab)
    glove_vectors, glove_info = None, None
    if glove_path is not None:
        vocab, glove_vectors, glove_info = load_glove_table(glove_path, vocab, glove_extra_words)
        print("GloVe:", json.dumps(glove_info))
    char_to_id = build_char_vocab(splits["train"])
    id_to_tag = {i: t for t, i in tag_to_id.items()}
    datasets = {
        name: Stage2Dataset(records, vocab, char_to_id, intent_to_id, tag_to_id)
        for name, records in splits.items()
    }
    freqs = word_frequencies(splits["train"], vocab)
    print("Character vocabulary size:", len(char_to_id))
    return {
        "splits": splits,
        "vocab": vocab,
        "char_to_id": char_to_id,
        "intent_to_id": intent_to_id,
        "tag_to_id": tag_to_id,
        "id_to_tag": id_to_tag,
        "datasets": datasets,
        "freqs": freqs,
        "train_vocab_size": train_vocab_size,
        "glove_vectors": glove_vectors,
        "glove_info": glove_info,
    }


def build_model(data, config):
    unk_prob = config["unk_z"] / (config["unk_z"] + data["freqs"])
    unk_prob[0] = 0.0  # <pad>
    unk_prob[1] = 0.0  # <unk>
    return JointBiGRUStage2(
        vocab_size=data["train_vocab_size"],
        glove_vectors=data["glove_vectors"],
        num_chars=len(data["char_to_id"]),
        num_intents=len(data["intent_to_id"]),
        id_to_tag=data["id_to_tag"],
        unk_prob=unk_prob,
        **{k: (tuple(config[k]) if k == "char_kernels" else config[k]) for k in MODEL_KEYS},
    )


def make_loaders(data, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(
            ds,
            batch_size=batch_size if name == "train" else 128,
            shuffle=(name == "train"),
            collate_fn=collate,
            generator=generator if name == "train" else None,
        )
        for name, ds in data["datasets"].items()
    }


def train_model(data, config, output_dir, checkpoint_name="best_bigru_stage2.pt", verbose=True):
    config = {**DEFAULT_CONFIG, **config}
    torch.set_num_threads(config["threads"])
    seed_everything(config["seed"])
    want = config.get("device", "cpu")
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(want)
    if verbose:
        print("Training device:", device,
              f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    loaders = make_loaders(data, config["batch_size"], config["seed"])
    model = build_model(data, config).to(device)
    total, trainable = count_parameters(model)
    if verbose:
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
    assert total <= 20_000_000, "Parameter budget (20M) exceeded"

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = [p for n, p in named if p.ndim < 2 or "norm" in n]
    decay = [p for n, p in named if not (p.ndim < 2 or "norm" in n)]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config["lr"],
    )
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * config["epochs"]
    warmup = steps_per_epoch * config["warmup_epochs"]

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / checkpoint_name

    history = []
    best = {"score": -1.0, "epoch": -1}
    run_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            result = model(*(batch[k].to(device) for k in
                             ("tokens", "chars", "mask", "tags", "intents")))
            result["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            optimizer.step()
            scheduler.step()
            loss_sum += result["loss"].item() * batch["tokens"].shape[0]
            seen += batch["tokens"].shape[0]

        dev = evaluate(model, loaders["dev"], device, data["id_to_tag"])
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "validation_loss": dev["loss"],
            "validation_intent_accuracy": dev["intent_accuracy"],
            "validation_slot_micro_f1": dev["slot_micro_f1"],
            "selection_score": dev["combined"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": elapsed,
        }
        history.append(row)
        improved = dev["combined"] > best["score"]
        if improved:
            best = {"score": dev["combined"], "epoch": epoch, **dev}
            torch.save(
                {
                    "model": trainable_state(model),
                    "config": config,
                    "vocab": data["vocab"],
                    "char_to_id": data["char_to_id"],
                    "intent_to_id": data["intent_to_id"],
                    "tag_to_id": data["tag_to_id"],
                    "epoch": epoch,
                    "validation_score": dev["combined"],
                    "validation_intent_accuracy": dev["intent_accuracy"],
                    "validation_slot_micro_f1": dev["slot_micro_f1"],
                    "total_parameters": total,
                    "trainable_parameters": trainable,
                },
                checkpoint_path,
            )
        if verbose:
            print(
                f"Epoch {epoch:2d}/{config['epochs']} | train loss {row['train_loss']:.4f} | "
                f"val loss {row['validation_loss']:.4f} | "
                f"val intent {row['validation_intent_accuracy']:.2f}% | "
                f"val slot F1 {row['validation_slot_micro_f1']:.2f}% | "
                f"combined {row['selection_score']:.2f} | {elapsed:.1f}s"
                + ("  <- best, saved" if improved else ""),
                flush=True,
            )
    total_time = time.perf_counter() - run_start
    with (output_dir / checkpoint_name.replace(".pt", "_history.json")).open("w") as f:
        json.dump({"config": config, "history": history, "best_epoch": best["epoch"],
                   "best_validation_score": best["score"], "total_seconds": total_time}, f, indent=2)
    if verbose:
        print(f"\nTotal training time: {total_time / 60:.1f} min")
        print(f"Best epoch: {best['epoch']}")
        print(
            f"Best validation score (combined): {best['score']:.2f} "
            f"(intent {best['intent_accuracy']:.2f}%, slot F1 {best['slot_micro_f1']:.2f}%)"
        )
        print(f"Checkpoint: {checkpoint_path}")
    return {"history": history, "best": best, "checkpoint": checkpoint_path,
            "total_parameters": total, "trainable_parameters": trainable}


def trainable_state(model):
    """State dict without the frozen GloVe table (rebuilt from the GloVe file)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("glove.")}


def load_checkpoint(path, data):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["vocab"] == data["vocab"] and saved["tag_to_id"] == data["tag_to_id"]
    model = build_model(data, saved["config"])
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    assert not unexpected and all(k.startswith("glove.") for k in missing)
    model.eval()
    return model, saved


def save_ensemble(member_checkpoints, data, path):
    """Bundle the selected member checkpoints into one ensemble checkpoint."""
    saved = [torch.load(p, map_location="cpu", weights_only=False) for p in member_checkpoints]
    torch.save({
        "members": [{k: s[k] for k in ("model", "config", "epoch", "validation_score")} for s in saved],
        "member_files": [str(p) for p in member_checkpoints],
        "vocab": data["vocab"],
        "char_to_id": data["char_to_id"],
        "intent_to_id": data["intent_to_id"],
        "tag_to_id": data["tag_to_id"],
        "glove_info": data["glove_info"],
    }, path)


def load_ensemble(path, data):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["vocab"] == data["vocab"] and saved["tag_to_id"] == data["tag_to_id"]
    members = []
    for entry in saved["members"]:
        model = build_model(data, entry["config"])
        missing, unexpected = model.load_state_dict(entry["model"], strict=False)
        assert not unexpected and all(k.startswith("glove.") for k in missing)
        members.append(model)
    ensemble = JointEnsemble(members)
    ensemble.eval()
    return ensemble, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("1.0/data/en-US.jsonl"))
    parser.add_argument("--output", default="stage2_enhanced_outputs")
    parser.add_argument("--checkpoint-name", default="best_bigru_stage2.pt")
    parser.add_argument("--config", type=json.loads, default={},
                        help='JSON overrides, e.g. \'{"epochs": 30, "hidden_size": 192}\'')
    parser.add_argument("--glove", type=Path, default=None,
                        help="optional GloVe .txt file (pretrained word vectors only)")
    parser.add_argument("--glove-extra-words", type=int, default=0)
    args = parser.parse_args()

    data = prepare(args.data, args.glove, args.glove_extra_words)
    run = train_model(data, args.config, args.output, args.checkpoint_name)
    model, _ = load_checkpoint(run["checkpoint"], data)
    loaders = make_loaders(data, 32, 0)
    for split in ["dev", "test"]:
        m = evaluate(model, loaders[split], "cpu", data["id_to_tag"])
        print(f"FINAL {split}: intent {m['intent_accuracy']:.2f}% | slot F1 {m['slot_micro_f1']:.2f}%")


if __name__ == "__main__":
    main()
