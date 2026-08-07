"""
Unit tests for DatasetSplitter.
"""

import os
import tempfile
import pytest
from slm.tokenizer.bpe import BPETokenizer
from slm.dataset.prep import DatasetSplitter


def test_dataset_splitter():
    splitter = DatasetSplitter(seed=42)
    docs = [f"Document number {i} for dataset splitting test." for i in range(100)]

    train, val, test = splitter.split_corpus(docs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = splitter.save_splits(train, val, test, output_dir=tmpdir)
        assert os.path.isfile(paths["train"])
        assert os.path.isfile(paths["val"])
        assert os.path.isfile(paths["test"])

        tokenizer = BPETokenizer()
        tokenizer.train_on_texts(train[:10], vocab_size=50)

        stats = splitter.estimate_token_counts(train[:10], tokenizer)
        assert stats["total_documents"] == 10
        assert stats["total_tokens"] > 0

        # Test binary token caching
        token_ids = [1, 2, 3, 10, 25, 42]
        bin_path = os.path.join(tmpdir, "tokens.bin")
        splitter.cache_token_ids(token_ids, bin_path)

        loaded_ids = splitter.load_cached_token_ids(bin_path)
        assert loaded_ids == token_ids
