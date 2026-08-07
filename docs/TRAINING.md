# Training Guide - Small Language Model (SLM)

## Training Overview

`lawslm` provides an industrial training pipeline supporting custom text domain datasets, subword BPE tokenization from scratch, mixed precision (FP16/BF16), gradient accumulation, and automated checkpointing.

## Quickstart Training

To launch training using the CLI:

```bash
python -m slm.cli.main train --config configs/nano_config.yaml --dataset data/custom_domain.txt
```

Or using the Python runner script:

```bash
python scripts/train_run.py
```

## Configurable Hyperparameters

Edit `configs/default_config.yaml` to adjust model capacity and training schedule:

```yaml
model:
  vocab_size: 32000
  d_model: 512
  n_heads: 8
  n_layers: 8
  d_ff: 2048
  max_seq_len: 1024
  norm_type: "rmsnorm"
  activation: "swiglu"
  pos_encoding_type: "rope"

training:
  batch_size: 16
  learning_rate: 3.0e-4
  grad_accum_steps: 2
  mixed_precision: "fp16"
  optimizer_name: "adamw"
  scheduler_name: "cosine"
```
