---
language:
- en
license: mit
tags:
- law
- slm
- causal-lm
- pytorch
- transformer
- text-generation
pipeline_tag: text-generation
inference: true
model_name: LawSLM
---

# Model Card — LawSLM (Small Language Model)


## Model Overview

- **Model Name**: LawSLM
- **Model Type**: Decoder-Only Transformer Language Model
- **Architecture**:
  - Positional Encoding: Rotary Position Embeddings (RoPE)
  - Normalization: Root Mean Square Normalization (RMSNorm)
  - Activation Function: Swish-Gated Linear Unit (SwiGLU)
  - Weight Tying: Enabled (`lm_head.weight == token_embeddings.weight`)
- **Language(s)**: English, Legal & Technical Domain Text
- **Framework**: PyTorch 2.6+ (Built 100% from Scratch)

---

## Intended Use

1. **Educational & Legal Information**: Plain-language explanations of legal statutes, terminology, and document templates.
2. **General AI Assistance**: Multi-turn dialogue, text summarization, and writing.
3. **Coding & Technical Querying**: Python, C++, Java, SQL, and Machine Learning concepts.

---

## Pipeline & Training Details

- **Tokenizer**: Custom Byte-Pair Encoding (BPE) Tokenizer with exact word-boundary space preservation (`\n` and `\S+` pattern preservation).
- **Optimizer**: Custom AdamW / Lion with Cosine Annealing Learning Rate Schedule and Warmup.
- **Precision**: FP32 / Mixed Precision AMP.
- **Evaluation Metrics**:
  - Cross-Entropy Validation Loss
  - Perplexity ($\exp(\text{Loss})$)
  - Top-1 and Top-5 Next-Token Prediction Accuracy
  - Inference Latency (ms/token and tokens/sec)

---

## Usage Instructions

### Interactive Console Chat
```bash
python -m slm.cli.main chat
```

### Full Automated Pipeline
```bash
python -m slm.cli.main pipeline --max_steps 300 --epochs 50
```

### REST API Server
```bash
uvicorn slm.api.app:app --host 0.0.0.0 --port 8000
```
- Endpoint `GET /system-prompt`: LawSLM System Prompt & Safety Guidelines.
- Endpoint `POST /generate`: Autoregressive Text Generation.
- Endpoint `POST /tokenizer/encode`: Token ID Encoding.
