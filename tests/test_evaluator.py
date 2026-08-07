"""
Unit tests for ModelEvaluator.
"""

import pytest
import torch
from slm.config.model_config import ModelConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.tokenizer.bpe import BPETokenizer
from slm.dataset.dataset import CausalLMDataset
from slm.dataset.loader import create_dataloader
from slm.evaluation.evaluator import ModelEvaluator


def test_model_evaluator():
    corpus = [
        "User: hello\nSLM: Hello! How can I help you today?\n",
        "User: what is python\nSLM: Python is a programming language.\n"
    ] * 5

    tokenizer = BPETokenizer()
    tokenizer.train_on_texts(corpus, vocab_size=100)

    config = ModelConfig(vocab_size=len(tokenizer.vocab), d_model=32, n_heads=2, n_layers=1, max_seq_len=64)
    model = SLMForCausalLM(config)

    dataset = CausalLMDataset(corpus, tokenizer, max_seq_len=64)
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    evaluator = ModelEvaluator(model, tokenizer, device="cpu")

    report = evaluator.full_evaluation_report(loader)

    assert "val_loss" in report
    assert "perplexity" in report
    assert "top_1_accuracy" in report
    assert "top_5_accuracy" in report
    assert "tokens_per_second" in report
    assert "ms_per_token" in report
    assert report["val_loss"] >= 0.0
    assert report["perplexity"] >= 1.0
