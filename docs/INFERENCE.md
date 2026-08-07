# Inference Guide - Small Language Model (SLM)

## Text Generation Engine

`lawslm` includes a complete autoregressive text sampling engine supporting:
- Temperature scaling
- Top-K filtering
- Top-P (Nucleus) sampling
- Repetition, Frequency, and Presence penalties
- Real-time streaming callbacks

## CLI Text Generation

```bash
python -m slm.cli.main generate \
  --prompt "The future of artificial intelligence" \
  --max_tokens 128 \
  --temperature 0.8 \
  --top_k 40 \
  --top_p 0.9
```

## Python API Usage

```python
from slm.config.model_config import ModelConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.tokenizer.bpe import BPETokenizer
from slm.sampling.generator import TextGenerator

# Load tokenizer and model
tokenizer = BPETokenizer.load("checkpoints/tokenizer")
model = SLMForCausalLM(ModelConfig())

generator = TextGenerator(model, tokenizer)
text = generator.generate("Once upon a time", max_new_tokens=100)
print(text)
```
