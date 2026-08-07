"""
DataLoader factory for training and validation pipelines.
"""

from typing import List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset

from slm.dataset.dataset import CausalLMDataset
from slm.tokenizer.bpe import BPETokenizer


def causal_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function combining sequence samples into padded batch tensors.

    Args:
        batch: List of (input_ids, target_ids) tuple pairs.

    Returns:
        Tuple of (input_batch [batch_size, seq_len], target_batch [batch_size, seq_len]).
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    input_batch = torch.stack(inputs, dim=0)
    target_batch = torch.stack(targets, dim=0)

    return input_batch, target_batch


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True
) -> DataLoader:
    """
    Constructs PyTorch DataLoader configured for causal LM training.

    Args:
        dataset: Target dataset instance.
        batch_size: Batch capacity.
        shuffle: Whether to shuffle data samples.
        num_workers: Multi-processing data worker count.
        drop_last: Whether to drop final incomplete batch.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=causal_collate_fn,
        drop_last=drop_last
    )
