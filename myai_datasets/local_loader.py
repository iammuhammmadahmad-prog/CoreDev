"""
Local text dataloader for train_v5.

Reads myai_datasets/processed/train.txt (falls back to every .txt in
myai_datasets/raw/), tokenizes with tiktoken, and yields endless random
(x, y) batches of shape (batch_size, seq_len).

Drop-in replacement for get_wiki_dataloader.
"""

from __future__ import annotations

from pathlib import Path

import tiktoken
import torch

DATA_DIR = Path(__file__).resolve().parent
PROCESSED = DATA_DIR / "processed" / "train.txt"
RAW_DIR = DATA_DIR / "raw"

# Must match the tokenizer your model config / checkpoints expect.
ENCODING_NAME = "gpt2"


def _read_text() -> str:
    if PROCESSED.exists() and PROCESSED.stat().st_size > 0:
        print(f"[data] Using {PROCESSED}")
        return PROCESSED.read_text(encoding="utf-8", errors="ignore")

    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No training text found in {PROCESSED} or {RAW_DIR}")
    print(f"[data] processed/train.txt missing, using raw files: {[f.name for f in files]}")
    return "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)


class LocalTextLoader:
    """Iterable that yields random batches forever."""

    def __init__(self, batch_size: int, seq_len: int, pin_memory: bool = False):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pin_memory = pin_memory

        enc = tiktoken.get_encoding(ENCODING_NAME)
        self.n_vocab = enc.n_vocab
        text = _read_text()
        ids = enc.encode_ordinary(text)
        self.tokens = torch.tensor(ids, dtype=torch.long)
        print(f"[data] {len(text):,} chars -> {len(self.tokens):,} tokens (vocab {self.n_vocab})")

        if len(self.tokens) < 2:
            raise ValueError("Training text is empty after tokenization.")
        # Tiny corpus: repeat it so at least one full window exists.
        if len(self.tokens) < seq_len + 1:
            reps = (seq_len + 1) // len(self.tokens) + 1
            self.tokens = self.tokens.repeat(reps)

    def __iter__(self):
        n = len(self.tokens) - self.seq_len - 1
        while True:
            starts = torch.randint(0, n + 1, (self.batch_size,))
            x = torch.stack([self.tokens[s : s + self.seq_len] for s in starts])
            y = torch.stack([self.tokens[s + 1 : s + 1 + self.seq_len] for s in starts])
            if self.pin_memory:
                x, y = x.pin_memory(), y.pin_memory()
            yield x, y


def get_local_dataloader(
    batch_size: int,
    seq_len: int,
    prefetch: int = 4,
    pin_memory: bool = False,
    worker_affinity=None,
) -> LocalTextLoader:
    # prefetch / worker_affinity accepted for compatibility; not needed for
    # an in-memory dataset.
    return LocalTextLoader(batch_size, seq_len, pin_memory)
