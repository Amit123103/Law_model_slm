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
        Tokenizes documents and builds sliding sequence windows for causal LM training.
        """
        all_token_ids: List[int] = []

        for doc in documents:
            tokens = self.tokenizer.encode(doc, add_special_tokens=True)
            all_token_ids.extend(tokens)

        # Build sequence windows of size (max_seq_len + 1)
        # target sequence y is input sequence x shifted right by 1
        seq_len = self.max_seq_len
        total_tokens = len(all_token_ids)

        if total_tokens <= seq_len:
            # Pad short document if needed
            pad_len = (seq_len + 1) - total_tokens
            all_token_ids = all_token_ids + [self.tokenizer.vocab.pad_id] * pad_len
            total_tokens = len(all_token_ids)

        for i in range(0, total_tokens - seq_len, self.stride):
            chunk = all_token_ids[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                input_seq = chunk[:-1]
                target_seq = chunk[1:]
                self.input_chunks.append(input_seq)
                self.target_chunks.append(target_seq)

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
