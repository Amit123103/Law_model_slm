"""
PyTorch Dataset implementations for Causal Language Modeling (standard and streaming).
"""

from typing import List, Tuple, Optional, Any, Dict, Iterator
import torch
from torch.utils.data import Dataset, IterableDataset

from slm.tokenizer.bpe import BPETokenizer
from slm.utils.logger import get_logger

logger = get_logger("slm.dataset")


class CausalLMDataset(Dataset):
    """
    Standard in-memory PyTorch Dataset for Causal Language Modeling (Decoder-only Transformer).
    Converts raw text documents into fixed-length chunked sequence tensors.
    """

    def __init__(
        self,
        documents: List[str],
        tokenizer: BPETokenizer,
        max_seq_len: int = 512,
        stride: Optional[int] = None
    ) -> None:
        """
        Initializes CausalLMDataset.

        Args:
            documents: List of text string documents.
            tokenizer: Trained BPETokenizer instance.
            max_seq_len: Sequence window context length.
            stride: Sliding window stride length (defaults to max_seq_len).
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len

        self.input_chunks: List[List[int]] = []
        self.target_chunks: List[List[int]] = []

        self._process_documents(documents)

    def _process_documents(self, documents: List[str]) -> None:
        """
        Tokenizes documents independently and builds sequence windows for causal LM training.
        """
        seq_len = self.max_seq_len
        pad_id = self.tokenizer.vocab.pad_id

        for doc in documents:
            doc_tokens = self.tokenizer.encode(doc, add_special_tokens=True)
            if not doc_tokens:
                continue

            if len(doc_tokens) <= seq_len:
                pad_len = (seq_len + 1) - len(doc_tokens)
                chunk = doc_tokens + [pad_id] * pad_len
                self.input_chunks.append(chunk[:-1])
                self.target_chunks.append(chunk[1:])
            else:
                for i in range(0, len(doc_tokens) - seq_len, self.stride):
                    chunk = doc_tokens[i:i + seq_len + 1]
                    if len(chunk) == seq_len + 1:
                        self.input_chunks.append(chunk[:-1])
                        self.target_chunks.append(chunk[1:])

        logger.info(f"Built CausalLMDataset: total sequence samples = {len(self.input_chunks)}")

    def __len__(self) -> int:
        return len(self.input_chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids tensor [seq_len], target_ids tensor [seq_len]).
        """
        input_ids = torch.tensor(self.input_chunks[idx], dtype=torch.long)
        target_ids = torch.tensor(self.target_chunks[idx], dtype=torch.long)
        return input_ids, target_ids


class StreamingCausalDataset(IterableDataset):
    """
    Memory-efficient PyTorch IterableDataset for streaming massive datasets line-by-line.
    """

    def __init__(
        self,
        file_paths: List[str],
        tokenizer: BPETokenizer,
        max_seq_len: int = 512
    ) -> None:
        """
        Initializes StreamingCausalDataset.
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Streams lines from text files, tokenizes, and yields causal (input, target) tensors.
        """
        buffer: List[int] = []

        for path in self.file_paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    tokens = self.tokenizer.encode(line.strip(), add_special_tokens=False)
                    buffer.extend(tokens)

                    while len(buffer) >= self.max_seq_len + 1:
                        chunk = buffer[:self.max_seq_len + 1]
                        buffer = buffer[self.max_seq_len:]
                        
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                        yield input_ids, target_ids
