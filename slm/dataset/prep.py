"""
Dataset Splitter, Token Count Estimator, and Binary Cacher for SLM datasets.
Handles train/validation/test splitting, disk-based token caching, and dataset metadata generation.
"""

import os
import json
import random
import array
from typing import List, Tuple, Dict, Any
from slm.tokenizer.bpe import BPETokenizer
from slm.utils.logger import get_logger

logger = get_logger("slm.dataset.prep")


class DatasetSplitter:
    """Splits, caches, and estimates token statistics for SLM datasets."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split_corpus(
        self,
        documents: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Shuffles and splits documents into train, val, and test sets according to ratios.
        """
        if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
            raise ValueError(f"Ratios must sum to 1.0. Got: {train_ratio + val_ratio + test_ratio}")

        shuffled = list(documents)
        random.seed(self.seed)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_docs = shuffled[:n_train]
        val_docs = shuffled[n_train:n_train + n_val]
        test_docs = shuffled[n_train + n_val:]

        logger.info(f"Split {n} documents into Train={len(train_docs)}, Val={len(val_docs)}, Test={len(test_docs)}")
        return train_docs, val_docs, test_docs

    def save_splits(self, train_docs: List[str], val_docs: List[str], test_docs: List[str], output_dir: str = "data/splits") -> Dict[str, str]:
        """Saves text splits to disk."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            "train": os.path.join(output_dir, "train.txt"),
            "val": os.path.join(output_dir, "val.txt"),
            "test": os.path.join(output_dir, "test.txt")
        }

        for key, p in paths.items():
            docs = train_docs if key == "train" else (val_docs if key == "val" else test_docs)
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n\n<|endoftext|>\n\n".join(docs))

        logger.info(f"Saved dataset splits to {output_dir}")
        return paths

    @staticmethod
    def estimate_token_counts(documents: List[str], tokenizer: BPETokenizer) -> Dict[str, Any]:
        """Estimates total token count and average tokens per document across a corpus."""
        total_tokens = 0
        doc_lengths = []

        for doc in documents:
            tokens = tokenizer.encode(doc, add_special_tokens=True)
            cnt = len(tokens)
            total_tokens += cnt
            doc_lengths.append(cnt)

        avg_tokens = (total_tokens / len(documents)) if documents else 0.0
        return {
            "total_documents": len(documents),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 2),
            "max_tokens_in_doc": max(doc_lengths) if doc_lengths else 0,
            "min_tokens_in_doc": min(doc_lengths) if doc_lengths else 0
        }

    @staticmethod
    def cache_token_ids(token_ids: List[int], filepath: str) -> str:
        """Saves token IDs as a compact binary array file (.bin)."""
        arr = array.array("I", token_ids)
        with open(filepath, "wb") as f:
            arr.tofile(f)
        logger.info(f"Cached {len(token_ids)} token IDs to {filepath}")
        return filepath

    @staticmethod
    def load_cached_token_ids(filepath: str) -> List[int]:
        """Loads token IDs from a compact binary array file (.bin)."""
        arr = array.array("I")
        with open(filepath, "rb") as f:
            arr.fromfile(f, os.path.getsize(filepath) // arr.itemsize)
        return arr.tolist()
