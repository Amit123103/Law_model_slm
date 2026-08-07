# Architectural Documentation - Small Language Model (SLM)

## System Architecture

`lawslm` is an industrial-grade, decoder-only Small Language Model (SLM) ecosystem built completely from scratch using Python and PyTorch tensor primitives.

```
+-----------------------------------------------------------------------+
|                             SLM ENGINE                                |
+-----------------------------------------------------------------------+
|  [REST API / FastAPI]                  [CLI Interface / main.py]      |
+-----------------------------------++----------------------------------+
                                    ||
+-----------------------------------vv----------------------------------+
|                           Text Generator                              |
|       (Greedy, Temp, Top-K, Top-P, Penalties, Streaming Callbacks)    |
+-----------------------------------++----------------------------------+
                                    ||
+-----------------------------------vv----------------------------------+
|                         SLMForCausalLM                                |
|  Token Embeddings -> RoPE -> N x TransformerBlock -> RMSNorm -> Head  |
+-----------------------------------++----------------------------------+
                                    ||
+-----------------------------------vv----------------------------------+
|                        Transformer Block                              |
|  Pre-RMSNorm -> Multi-Head Causal Attention -> Pre-RMSNorm -> SwiGLU  |
+-----------------------------------------------------------------------+
```

## Modular Breakdown

1. **Tokenizer (`slm/tokenizer/`)**:
   - Pure Python Byte-Pair Encoding (BPE) subword algorithm (`bpe.py`).
   - Character tokenizer fallback (`char_tokenizer.py`).
   - Vocabulary manager, frequency dictionary, and JSON serialization (`vocab.py`).

2. **Dataset & Cleaning (`slm/dataset/`)**:
   - Text cleaning, NFC Unicode normalization, HTML stripping, deduplication (`cleaner.py`).
   - Ingestion parsers for TXT, CSV, JSON, JSONL, MD, HTML, XML (`readers.py`).
   - Sliding causal context window PyTorch Dataset (`dataset.py`).

3. **Embeddings & Normalization (`slm/embeddings/`, `slm/normalization/`)**:
   - Rotary Position Embeddings (RoPE) applied to Queries and Keys (`positional.py`).
   - Learnable and Sinusoidal positional embeddings option (`positional.py`).
   - Root Mean Square Layer Normalization (`rmsnorm.py`).
   - Layer Normalization (`layernorm.py`).

4. **Attention & FeedForward (`slm/attention/`, `slm/feedforward/`)**:
   - Scaled Dot-Product Attention with triangular causal mask (`causal_attention.py`).
   - Multi-Head Causal Attention with RoPE (`causal_attention.py`).
   - SwiGLU (Swish Gated Linear Unit) Feed-Forward Network (`mlp.py`).

5. **Optimizers & Schedulers (`slm/optimizer/`, `slm/scheduler/`)**:
   - Custom `AdamW` with decoupled weight decay (`adamw.py`).
   - Custom `Lion` (EvoLved Sign Momentum) optimizer (`lion.py`).
   - Cosine Annealing with Warmup learning rate scheduler (`schedulers.py`).

6. **Training Engine (`slm/training/`)**:
   - Teacher-forcing training loop with AMP FP16/BF16, gradient accumulation, gradient clipping, evaluation, and checkpoint manager integration.
