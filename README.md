# CoreDev: A Language Model Built From Scratch

CoreDev is a from-scratch language model stack: tokenizer, transformer, trainer, RAG, CLI, and a local Studio UI. It is written in PyTorch, the network weights start random, and no pretrained model is downloaded. The v5 models reuse the GPT-2 vocabulary through `tiktoken`; everything else is this repository's own code.

## Current status

The current model is **v5 tiny, 19.2M parameters**, trained on a laptop in two stages:

| Stage | Data | Steps | Average loss |
|---|---|---|---|
| 1. Pretrain | Wikipedia | up to a 100,000-step budget | 5.6 |
| 2. Continue | Shakespeare text | 19,000 | 2.2 |

The two losses are measured on different datasets and are not comparable. Shakespeare is a small, stylistically narrow corpus, so a lower loss there does not mean a better model.

**What it can do:** it writes Shakespeare-style text. Because it was trained last on Shakespeare, it continues in that style whatever the prompt says.

**What it cannot do:** answer questions or follow instructions. At ~19M parameters with a 50k vocabulary, samples can look fluent and still be empty or circular. It is a language-modelling project, not a chat assistant.

## Stack

| Layer | What it is |
|---|---|
| Tokenizer | Custom BPE for v1; GPT-2 `tiktoken` (50,257 tokens) for Wikipedia and v5 |
| Model | Decoder-only GPT. v1–v4 are classic (learned positional embeddings, LayerNorm, GELU). v5 is modern (RoPE, RMSNorm, SwiGLU), with support for grouped-query attention and a KV cache |
| Train | Local text, Wikipedia stream, QA fine-tune, cosine LR schedule, checkpoint resume |
| Inference | Greedy, temperature, top-k, top-p |
| Memory | TF-IDF RAG over a local knowledge base |
| Apps | React Studio at `http://127.0.0.1:8765`, plus a Tk desktop app |
| CLI | `prepare` → `train` / `train-v5` → `finetune-qa` → `evaluate` → `generate` → `studio` |

## Quick start

Requires Python 3.10+ and a virtualenv:

```bash
git clone https://github.com/iammuhammmadahmad-prog/Rija.git
cd Rija
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py smoke
python main.py studio
```

Studio opens at [http://127.0.0.1:8765](http://127.0.0.1:8765). Load a `.pt` checkpoint from the sidebar and type a prompt. Studio streams tokens from the local checkpoint, so your prompts never leave your machine.

## Train

**Tiny v1** (custom BPE, CPU, a few minutes):

```bash
python main.py prepare
python main.py train --preset tiny --steps 1000 --version v1
python main.py generate "Machine learning" --strategy top_p --version v1
```

**v5 Wikipedia pretrain** (tiktoken, laptop-safe tiny preset):

```bash
python -m trainer.train_v5 --size tiny --seq-len 256 --batch-size 8 --no-compile --no-resume
# or
python main.py train-v5 --size tiny --seq-len 256 --batch-size 8
```

Resume is the default. Checkpoints are saved to `checkpoints/v5_tiny/`. Watch the log with `tail -f logs/train_v5_tiny.log`.

<!-- TODO: add the exact command used for the Shakespeare stage and where the Shakespeare text file comes from. -->

**QA fine-tune** on a finished base:

```bash
python main.py finetune-qa --base checkpoints/v5_tiny/latest.pt --out checkpoints/v5_qa
python main.py generate "What is a transformer?" --qa --rag --version v5_qa
```

## v5 size ladder

| Size | Params (approx) | Hardware |
|---|---|---|
| `tiny` | ~19M | Laptop CPU or GPU |
| `small` | ~50–80M | Laptop GPU, 8–12 GB |
| `base` | ~125M | Single 16–24 GB GPU |
| `medium` | ~350M | 24–48 GB GPU |
| `large` / `xl` / `chatgpt` | 1.5B–175B | Blocked unless `--force-huge` |

`tiny` is the tested path. Larger sizes are configurations only, with no promise that a laptop can train them.

## Studio and desktop

```bash
python main.py studio --host 127.0.0.1 --port 8765
python main.py gui
```

The desktop app has Chat, Trainer, Models, Tools, and Logs tabs.

## CLI

```text
python main.py smoke         # tokenizer + tiny forward pass
python main.py prepare       # clean raw text, train BPE
python main.py train         # v1 loop
python main.py train-v5      # modern GPT scale ladder
python main.py finetune-qa   # instruction / Q&A pass
python main.py seed-facts    # load qa_seed.jsonl into RAG
python main.py generate      # sample a checkpoint
python main.py evaluate      # perplexity / accuracy
python main.py studio        # web playground
python main.py gui           # Tk desktop
```

## Layout

```text
Rija/
├── main.py                 # CLI
├── generate.py             # interactive generation
├── tokenizer/              # custom BPE
├── model/                  # GPT (classic + modern)
│   ├── attention.py        # multi-head attention, GQA, KV cache
│   ├── rope.py             # rotary positional embeddings
│   ├── norms.py            # RMSNorm
│   └── scaling.py          # v5 size presets
├── trainer/
│   ├── train.py            # v1
│   ├── train_wiki.py       # Wikipedia (classic)
│   ├── train_v5.py         # Wikipedia (modern)
│   ├── finetune_qa.py
│   └── cpu_runtime.py      # laptop P/E-core pinning
├── myai_datasets/          # raw / processed / Wikipedia loader / QA seed
├── inference/              # decode, RAG, checkpoint load
├── memory/                 # knowledge base, chat history, search
├── tools/                  # files, calculator, code runner
├── evaluation/
├── ui/
│   ├── app.py              # desktop
│   ├── studio.py           # FastAPI + SSE
│   └── static/             # React Studio
├── checkpoints/
└── tests/
```

## Tests

```bash
python -m pytest tests/ -v
```

## Known limitations

- Trained on a small amount of data at ~19M parameters, so it does not follow instructions.
- Sampled text can be fluent but empty or circular.
- The current checkpoint is biased toward Shakespeare because it was trained on it last.

## Next steps

- Train a smaller custom vocabulary (4k–8k tokens) on the same data. Of the three upgrades, this is likely to help most at this size.
- Train on more tokens with the same architecture, not a larger preset on CPU.
- Instruction fine-tuning on a cleaner dataset, so the model answers instead of only continuing text.
