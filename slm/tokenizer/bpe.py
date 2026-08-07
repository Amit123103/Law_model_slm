"""
Byte-Pair Encoding (BPE) Tokenizer implemented completely from scratch.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Union, Any
import torch

from slm.tokenizer.vocab import Vocabulary
from slm.utils.logger import get_logger

logger = get_logger("slm.tokenizer.bpe")


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer implemented from scratch.
    Handles subword tokenization, vocabulary merging, padding, and causal masking.
    """

    def __init__(self, vocab: Optional[Vocabulary] = None) -> None:
        """
        Initializes BPE Tokenizer.
        """
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

    def _get_stats(self, word_counts: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """
        Counts adjacent symbol pair frequencies across corpus word split tuples.
        """
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """
        Merges instances of `pair` in symbol tuple `word`.
        """
        first, second = pair
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train_on_texts(
        self,
        texts: List[str],
        vocab_size: int = 32000,
        min_frequency: int = 2
    ) -> None:
        """
        Trains BPE subword merge rules on input list of texts until target vocabulary size is met.

        Args:
            texts: Training text corpus.
            vocab_size: Target total vocabulary size.
            min_frequency: Minimum pair frequency required for merge operation.
        """
        logger.info(f"Training BPE Tokenizer targeting vocab_size={vocab_size}...")

        # 1. Build initial word counts with character-level representation + end-of-word marker
        import re
        word_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for text in texts:
            words = re.findall(r"\n|\S+", text)
            for word in words:
                if not word:
                    continue
                # Character representation with '</w>' suffix marking word boundary
                char_tuple = tuple(list(word[:-1]) + [word[-1] + "</w>"])
                word_counts[char_tuple] += 1

        # 2. Add base characters to vocabulary
        unique_chars: Set[str] = set()
        for word_tuple in word_counts.keys():
            for symbol in word_tuple:
                unique_chars.add(symbol)

        for char in sorted(list(unique_chars)):
            self.vocab.add_token(char)

        # 3. Iteratively learn merge pairs
        num_merges = vocab_size - len(self.vocab)
        logger.info(f"Base vocabulary size: {len(self.vocab)}. Performing up to {num_merges} merges...")

        for merge_idx in range(num_merges):
            pairs = self._get_stats(word_counts)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                logger.info(f"Stopping merges early at step {merge_idx}: max pair frequency < {min_frequency}")
                break

            merged_token = best_pair[0] + best_pair[1]
            self.vocab.add_token(merged_token, count=pairs[best_pair])

            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_idx

            # Apply merge to word_counts corpus
            new_word_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
            for word, count in word_counts.items():
                merged_word = self._merge_word(word, best_pair)
                new_word_counts[merged_word] += count
            word_counts = new_word_counts

        logger.info(f"BPE Tokenizer training complete! Final vocab size: {len(self.vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Applies learned BPE merge rules to a single word string.
        """
        if not word:
            return []

        symbols = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        
        while len(symbols) > 1:
            # Find candidate pairs present in word
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # Rank pairs by learned BPE merge order
            candidate_pairs = [p for p in pairs if p in self.bpe_ranks]

            if not candidate_pairs:
                break

            best_pair = min(candidate_pairs, key=lambda p: self.bpe_ranks[p])
            symbols = self._merge_word(symbols, best_pair)

        return list(symbols)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True
    ) -> List[int]:
        """
        Encodes string text into a list of integer token IDs.

        Args:
            text: Input string text.
            add_special_tokens: Prepend BOS and append EOS tokens.
            max_length: Optional target maximum sequence length.
            padding: Pad sequence with PAD token if below max_length.
            truncation: Truncate sequence if exceeding max_length.

        Returns:
            List of integer token IDs.
        """
        import re
        words = re.findall(r"\n|\S+", text)
        subwords: List[str] = []

        for word in words:
            subwords.extend(self._tokenize_word(word))

        token_ids = [self.vocab.get_id(sw) for sw in subwords]

        if add_special_tokens:
            token_ids = [self.vocab.bos_id] + token_ids + [self.vocab.eos_id]

        if max_length is not None:
            if len(token_ids) > max_length and truncation:
                token_ids = token_ids[:max_length]
            elif len(token_ids) < max_length and padding:
                token_ids = token_ids + [self.vocab.pad_id] * (max_length - len(token_ids))

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes sequence of token IDs back into string text.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: Filter out BOS, EOS, PAD, MASK tokens.

        Returns:
            Decoded string text.
        """
        special_ids = {self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id, self.vocab.mask_id}
        subwords: List[str] = []

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            token_str = self.vocab.get_token(tid)
            subwords.append(token_str)

        raw_text = "".join(subwords)
        decoded_text = raw_text.replace("\n</w>", "\n").replace("</w>", " ").strip()
        return decoded_text

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of strings into PyTorch tensors (input_ids and attention_mask).

        Returns:
            Tuple of (input_ids tensor [batch_size, seq_len], attention_mask tensor [batch_size, seq_len]).
        """
        encoded_list = [
            self.encode(
                t,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=False,
                truncation=truncation
            )
            for t in texts
        ]

        if max_length is None:
            max_len = max(len(seq) for seq in encoded_list)
        else:
            max_len = max_length

        batch_ids = []
        batch_masks = []

        for seq in encoded_list:
            if len(seq) > max_len and truncation:
                seq = seq[:max_len]
            pad_len = max_len - len(seq)
            
            padded_seq = seq + [self.vocab.pad_id] * pad_len
            attn_mask = [1] * len(seq) + [0] * pad_len

            batch_ids.append(padded_seq)
            batch_masks.append(attn_mask)

        return (
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_masks, dtype=torch.long)
        )

    def get_compression_ratio(self, text: str) -> Dict[str, float]:
        """
        Calculates character and byte compression statistics for a given text.
        """
        encoded_ids = self.encode(text, add_special_tokens=False)
        num_tokens = len(encoded_ids)
        num_chars = len(text)
        num_bytes = len(text.encode("utf-8"))

        return {
            "num_tokens": float(num_tokens),
            "num_chars": float(num_chars),
            "num_bytes": float(num_bytes),
            "char_compression_ratio": num_chars / max(1, num_tokens),
            "byte_compression_ratio": num_bytes / max(1, num_tokens),
        }

    def save(self, directory: str) -> None:
        """
        Saves vocabulary and learned BPE merge rules into target directory.
        """
        os.makedirs(directory, exist_ok=True)
        vocab_path = os.path.join(directory, "vocab.json")
        merges_path = os.path.join(directory, "merges.txt")

        self.vocab.save(vocab_path)

        with open(merges_path, "w", encoding="utf-8") as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

        logger.info(f"Saved BPE tokenizer state to {directory}")

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        """
        Loads BPE Tokenizer from saved directory containing vocab.json and merges.txt.
        """
        vocab_path = os.path.join(directory, "vocab.json")
        merges_path = os.path.join(directory, "merges.txt")

        vocab = Vocabulary.load(vocab_path)
        tokenizer = cls(vocab=vocab)

        merges = []
        bpe_ranks = {}
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split(" ")
                    if len(parts) == 2:
                        pair = (parts[0], parts[1])
                        merges.append(pair)
                        bpe_ranks[pair] = idx

        tokenizer.merges = merges
        tokenizer.bpe_ranks = bpe_ranks
        return tokenizer
