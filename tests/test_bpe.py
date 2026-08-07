"""
Unit tests for Byte-Pair Encoding (BPE) Tokenizer implemented from scratch.
"""

import os
import tempfile
import pytest

from slm.tokenizer.bpe import BPETokenizer
from slm.tokenizer.vocab import Vocabulary


def test_vocab_basic():
    vocab = Vocabulary()
    assert len(vocab) == 5  # <pad>, <unk>, <s>, </s>, <mask >
    idx = vocab.add_token("hello")
    assert vocab.get_token(idx) == "hello"
    assert vocab.get_id("hello") == idx
    assert vocab.get_id("nonexistent_token_xyz") == vocab.unk_id


def test_bpe_training_and_encode_decode():
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox is quick and fast",
    ] * 5

    tokenizer = BPETokenizer()
    tokenizer.train_on_texts(texts, vocab_size=50, min_frequency=1)

    assert len(tokenizer.vocab) <= 50

    test_str = "the quick brown fox"
    token_ids = tokenizer.encode(test_str, add_special_tokens=True)
    assert len(token_ids) > 0
    assert token_ids[0] == tokenizer.vocab.bos_id
    assert token_ids[-1] == tokenizer.vocab.eos_id

    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert decoded.strip() == test_str.strip()


def test_bpe_save_load():
    texts = ["hello world from scratch bpe tokenizer"] * 3
    tokenizer = BPETokenizer()
    tokenizer.train_on_texts(texts, vocab_size=20, min_frequency=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save(tmpdir)
        loaded_tok = BPETokenizer.load(tmpdir)

        assert len(loaded_tok.vocab) == len(tokenizer.vocab)
        encoded_orig = tokenizer.encode("hello world")
        encoded_loaded = loaded_tok.encode("hello world")
        assert encoded_orig == encoded_loaded
