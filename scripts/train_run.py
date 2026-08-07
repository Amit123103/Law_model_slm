"""
Script to launch SLM model training with custom or synthetic data.
"""

import sys
import os

# Ensure package root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slm.config.config_loader import load_config
from slm.dataset.cleaner import TextCleaner
from slm.dataset.dataset import CausalLMDataset
from slm.dataset.loader import create_dataloader
from slm.tokenizer.bpe import BPETokenizer
from slm.model.transformer_lm import SLMForCausalLM
from slm.training.trainer import Trainer
from slm.utils.logger import get_logger
from slm.utils.utils import set_seed

logger = get_logger("slm.scripts.train")


def run() -> None:
    config_path = os.path.join("configs", "nano_config.yaml")
    logger.info(f"Loading configuration from {config_path}...")
    model_config, train_config = load_config(config_path)

    set_seed(train_config.seed)

    # Sample corpus for demonstration
    raw_texts = [
        "Language models are trained to predict the next token given a context sequence of preceding tokens.",
        "The Small Language Model (SLM) ecosystem is designed for efficient training on custom text domain datasets.",
        "Transformer decoder blocks combine multi-head causal attention with rotary position embeddings and SwiGLU activations.",
        "Optimizers like AdamW and Lion enable fast convergence and stability when training deep neural networks.",
        "Byte Pair Encoding learns frequent subword merge rules directly from text without needing external NLP libraries."
    ] * 50

    cleaner = TextCleaner()
    cleaned_texts = [cleaner.clean_text(t) for t in raw_texts]

    # Train BPE Tokenizer from scratch
    tokenizer = BPETokenizer()
    tokenizer.train_on_texts(cleaned_texts, vocab_size=model_config.vocab_size)

    # Build dataset and dataloader
    dataset = CausalLMDataset(cleaned_texts, tokenizer, max_seq_len=model_config.max_seq_len)
    loader = create_dataloader(dataset, batch_size=train_config.batch_size, shuffle=True)

    # Initialize model
    model = SLMForCausalLM(model_config)

    # Initialize trainer
    trainer = Trainer(model, train_config, loader, tokenizer=tokenizer)

    # Run training loop
    summary = trainer.train()
    logger.info(f"Training completed successfully! Output summary: {summary}")


if __name__ == "__main__":
    run()
