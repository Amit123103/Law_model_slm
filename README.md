# Law_model_slm

# Implementation Plan - Build a Small Language Model (SLM) Completely From Scratch

Build an industrial-grade, decoder-only Small Language Model (SLM) ecosystem from total zero in Python and PyTorch (using only PyTorch tensor/autograd operations, NumPy, and standard Python libraries). No Hugging Face libraries (`transformers`, `tokenizers`, `datasets`, `accelerate`), no pre-built transformer blocks, and no pre-trained weights will be used.

## Architecture Overview

```
                        ┌──────────────────────────────────────────┐
                        │             SLM Engine CLI & API         │
                        └────────────────────┬─────────────────────┘
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      │                                      │                                      │
┌─────▼──────┐                       ┌───────▼────────┐                     ┌───────▼────────┐
│ Datasets & │                       │ Decoder-Only   │                     │ Trainer &      │
│ Pipelines  ├────────────┐          │ Transformer    ├────────────┐        │ Schedulers     │
└────────────┘            │          └───────▲────────┘            │        └────────────────┘
                          ▼                  │                     ▼
                   ┌──────────────┐          │             ┌──────────────┐
                   │ Custom BPE   ├──────────┘             │ Generation & │
                   │ Tokenizer    │                        │ Samplers     │
                   └──────────────┘                        └──────────────┘
```

The architecture strictly follows SOLID principles, clean architecture, explicit type annotations, docstrings, modularity, and zero reliance on third-party NLP libraries.

---

## User Review Required

> [!IMPORTANT]
> **Zero External NLP Dependencies**: The entire tokenizer (Byte-Pair Encoding), attention mechanism (Scaled Dot-Product & Multi-Head Causal Attention with RoPE / Sinusoidal / Learnable encodings), Normalization (RMSNorm & LayerNorm), FeedForward (SwiGLU & GELU), Decoder Block, Custom Optimizers (AdamW & Lion), Schedulers, Samplers (Top-K, Top-P, Repetition Penalty), Metrics (Perplexity, BLEU, ROUGE), Trainer, CLI, and API are implemented entirely from pure Python, PyTorch tensor primitive operations, and standard libraries.
> 
> **PyTorch Scope**: We will use PyTorch for tensor computation, autograd (`torch.Tensor`, `torch.nn.Module`, `torch.autograd`), CUDA operations, and AMP (`torch.cuda.amp.autocast`). Higher-level NLP abstractions are strictly hand-written.

---

## Key Components & File Breakdown

### 1. Configuration & Core Systems (`slm/config/`)
- `slm/config/model_config.py`: Dataclass config for Model (vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, norm_type, activation, rope, dropout).
- `slm/config/train_config.py`: Dataclass config for Training (batch_size, learning_rate, weight_decay, warmup_steps, max_steps, grad_accum_steps, clip_grad_norm, mixed_precision, device).
- `slm/config/config_loader.py`: YAML/JSON parser, environment variable overrides, validation.

### 2. Tokenizer from Scratch (`slm/tokenizer/`)
- `slm/tokenizer/bpe.py`: Pure Python Byte-Pair Encoding (BPE) implementation. Learns merge rules from raw text corpora, handles byte/character fallback, special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`, `<mask >`), serialization to JSON/text, encoding, decoding, padding, and causal mask generation.
- `slm/tokenizer/char_tokenizer.py`: Character-level fallback tokenizer.
- `slm/tokenizer/vocab.py`: Vocabulary manager, frequency builder, vocabulary pruning, compression stats.

### 3. Dataset & Data Processing (`slm/dataset/`)
- `slm/dataset/cleaner.py`: Text normalization, Unicode NFC normalization, HTML/XML tag removal, deduplication, sentence splitting, chunking.
- `slm/dataset/readers.py`: File ingestion for TXT, CSV, JSON, JSONL, Markdown, HTML, XML, with hooks for PDF text extraction.
- `slm/dataset/dataset.py`: PyTorch `Dataset` & `IterableDataset` (streaming/memory-efficient text chunks), token windowing, causal target shifting ($y_t = x_{t+1}$).
- `slm/dataset/loader.py`: DataLoader constructor with dynamic padding, batching, dynamic shuffling, data augmentation hooks.

### 4. Embeddings & Positional Encodings (`slm/embeddings/`)
- `slm/embeddings/token_embeddings.py`: Learnable token lookup matrix with optional weight-tying hook.
- `slm/embeddings/positional.py`: 
  - `LearnedPositionalEmbedding`: Trainable 1D position embeddings.
  - `SinusoidalPositionalEmbedding`: Fixed sine/cosine positional encodings.
  - `RotaryPositionEmbedding` (RoPE): Relative rotary position embeddings for query and key states.

### 5. Normalization & Feed-Forward (`slm/normalization/`, `slm/feedforward/`)
- `slm/normalization/rmsnorm.py`: Custom `RMSNorm` module ($y = \frac{x}{\sqrt{\text{Mean}(x^2) + \epsilon}} \cdot \gamma$).
- `slm/normalization/layernorm.py`: Custom `LayerNorm` module ($y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$).
- `slm/feedforward/mlp.py`: Standard MLP (GELU/ReLU) and SwiGLU ($x \text{Swish}(W_g x) \cdot (W_v x)$).

### 6. Attention & Transformer Block (`slm/attention/`, `slm/transformer/`)
- `slm/attention/causal_attention.py`: `ScaledDotProductAttention` with triangular causal mask, attention dropout, and attention weights return option. `MultiHeadAttention` splitting queries, keys, values across $h$ heads with RoPE or absolute positional encodings.
- `slm/transformer/block.py`: `TransformerBlock` combining Pre-LN/Pre-RMSNorm, Causal MHA, Residual Connections, and SwiGLU/MLP with gradient checkpointing hooks.
- `slm/transformer/model.py`: `SLMForCausalLM` full decoder architecture: Token Embeddings -> Positional Embeddings/RoPE -> $N \times$ Transformer Blocks -> Final Normalization -> LM Head Projection to Vocab. Supports weight tying between token embedding and LM head.

### 7. Optimizers & Schedulers (`slm/optimizer/`, `slm/scheduler/`)
- `slm/optimizer/adamw.py`: Pure PyTorch custom `AdamW` optimizer with decoupled weight decay ($m_t, v_t$ bias correction, parameters filtering).
- `slm/optimizer/lion.py`: Custom `Lion` optimizer (EvoLved Sign Momentum).
- `slm/scheduler/schedulers.py`: Custom `CosineWithWarmup`, `LinearWithWarmup`, `OneCycleLR` learning rate schedulers.

### 8. Training & Checkpointing (`slm/training/`, `slm/checkpoint/`)
- `slm/training/trainer.py`: Comprehensive `Trainer` supporting:
  - Mini-batch loop with Teacher Forcing
  - Gradient Accumulation ($k$ steps)
  - Mixed Precision Training (`torch.cuda.amp.autocast` + `GradScaler`)
  - Gradient Clipping (`clip_grad_norm_`)
  - Loss, Learning Rate, GPU Memory, and Gradient Norm tracking
  - TensorBoard & CSV Loggers
  - Early Stopping & Validation evaluation
- `slm/checkpoint/manager.py`: Checkpoint saving and resumption of model weights, optimizer state, scheduler state, tokenizer vocab, config, step/epoch count, random seed state.

### 9. Inference & Text Generation (`slm/sampling/`)
- `slm/sampling/generator.py`: `TextGenerator` with inference strategies:
  - Greedy decoding
  - Temperature scaling
  - Top-K filtering
  - Top-P (Nucleus) sampling
  - Typical sampling
  - Repetition, Frequency, and Presence penalties
  - Minimum probability thresholding
  - Stop sequence detection & streaming output callback

### 10. Evaluation & Benchmarking (`slm/evaluation/`, `slm/benchmarks/`)
- `slm/evaluation/metrics.py`: Cross-entropy, Perplexity ($e^{\text{loss}}$), Token-level accuracy, custom BLEU-4, custom ROUGE-L implementations.
- `slm/benchmarks/benchmark.py`: Benchmark suite for token throughput (tokens/sec), inference latency (ms), training iteration speed, VRAM/RAM memory consumption across CPU and CUDA devices.

### 11. API, CLI, Tests, Docker & Documentation
- `slm/api/app.py`: FastAPI server with endpoints: `/health`, `/info`, `/generate`, `/train`, `/eval`, `/tokenizer/encode`, `/tokenizer/decode`, `/config`.
- `slm/cli/main.py`: Full Click/Argparse CLI (`train`, `generate`, `tokenize`, `benchmark`, `eval`, `preprocess`).
- `tests/`: Extensive unit & integration test suite covering tokenizer, embeddings, attention, RMSNorm, model forward pass, loss compute, custom AdamW, trainer step, sampler output, API endpoints.
- `docker/`: `Dockerfile`, `docker-compose.yml` configured for GPU and CPU execution.
- `docs/`: Comprehensive guides (`README.md`, `ARCHITECTURE.md`, `TRAINING_GUIDE.md`, `INFERENCE_GUIDE.md`, `API_GUIDE.md`).

---

## Proposed Directory Hierarchy

```
c:/Users/amita/myprojects/lawslm/
├── configs/
│   ├── default_config.yaml
│   └── nano_config.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── INFERENCE.md
│   └── TRAINING.md
├── scripts/
│   ├── benchmark_run.py
│   ├── train_run.py
│   └── preprocess_data.py
├── slm/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── attention/
│   │   ├── __init__.py
│   │   └── causal_attention.py
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── model_config.py
│   │   └── train_config.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   └── readers.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── positional.py
│   │   └── token_embeddings.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   └── metrics.py
│   ├── feedforward/
│   │   ├── __init__.py
│   │   └── mlp.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── transformer_lm.py
│   ├── normalization/
│   │   ├── __init__.py
│   │   ├── layernorm.py
│   │   └── rmsnorm.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── adamw.py
│   │   └── lion.py
│   ├── sampling/
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── schedulers.py
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── bpe.py
│   │   ├── char_tokenizer.py
│   │   └── vocab.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── transformer/
│   │   ├── __init__.py
│   │   └── block.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_bpe.py
│   ├── test_dataset.py
│   ├── test_embeddings.py
│   ├── test_model.py
│   ├── test_normalization.py
│   ├── test_optimizer.py
│   ├── test_sampling.py
│   └── test_trainer.py
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Verification Plan

### Automated Testing (`pytest`)
- Run unit test suite: `pytest -v tests/`
- Verify tokenization roundtrip: `encode -> decode` matches original input.
- Verify causal attention masking: past tokens cannot attend to future tokens.
- Verify RMSNorm and LayerNorm output shape and gradient flow.
- Verify weight-tying consistency between token embeddings and LM head projection.
- Verify forward pass shape `[batch, seq_len, vocab_size]` and loss computation.
- Verify custom AdamW step updates parameters without NaNs.
- Verify Trainer execution over synthetic dataset for 5 iterations.
- Verify sampling methods (greedy, top-k, top-p) generate valid token sequences.

### Manual Verification
- Execute full training loop on sample text data using `python scripts/train_run.py`.
- Benchmark model speed and memory with `python scripts/benchmark_run.py`.
- Launch FastAPI server with `uvicorn slm.api.app:app` and test `/generate` endpoint.
