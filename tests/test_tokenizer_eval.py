"""
Unit tests for TokenizerEvaluator.
"""

import pytest
from slm.tokenizer.bpe import BPETokenizer
from slm.tokenizer.evaluator import TokenizerEvaluator


def test_tokenizer_evaluator():
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python and PyTorch for Small Language Models."
    ] * 5

    tokenizer = BPETokenizer()
    tokenizer.train_on_texts(texts, vocab_size=100)

    evaluator = TokenizerEvaluator(tokenizer)
    report = evaluator.full_evaluation(texts[:2])

    assert "vocab_size" in report
    assert "roundtrip" in report
    assert "compression" in report
    assert report["roundtrip"]["roundtrip_accuracy"] > 90.0
    assert report["compression"]["bytes_per_token"] > 0
    assert report["compression"]["unk_percentage"] == 0.0
