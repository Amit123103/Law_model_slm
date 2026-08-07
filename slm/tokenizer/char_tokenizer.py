"""
Character-level tokenizer implemented from scratch.
"""

from typing import List, Dict, Union, Optional, Tuple
import torch

from slm.tokenizer.vocab import Vocabulary


class CharTokenizer:
    """
    Character Tokenizer maps individual characters to token IDs and back.
    """

    def __init__(self, vocab: Optional[Vocabulary] = None) -> None:
        """
        Initializes Character Tokenizer with given or new Vocabulary.
        """
        self.vocab = vocab if vocab is not None else Vocabulary()

    def train_on_text(self, text: str) -> None:
        """
        Builds vocabulary from raw input string by extracting unique characters.

        Args:
            text: Input corpus text.
        """
        for char in text:
            self.vocab.add_token(char)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True
    ) -> List[int]:
        """
        Encodes input string into a list of integer character token IDs.

        Args:
            text: Input string.
            add_special_tokens: Whether to prepend BOS and append EOS tokens.
            max_length: Max allowed token sequence length.
            padding: Whether to pad output to max_length.
            truncation: Whether to truncate sequence exceeding max_length.

        Returns:
            List of integer token IDs.
        """
        tokens = [self.vocab.get_id(char) for char in text]

        if add_special_tokens:
            tokens = [self.vocab.bos_id] + tokens + [self.vocab.eos_id]

        if max_length is not None:
            if len(tokens) > max_length and truncation:
                tokens = tokens[:max_length]
            elif len(tokens) < max_length and padding:
                tokens = tokens + [self.vocab.pad_id] * (max_length - len(tokens))

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a sequence of character token IDs back to a string.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, filters out BOS, EOS, PAD, and MASK.

        Returns:
            Decoded string.
        """
        chars = []
        special_ids = {self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id, self.vocab.mask_id}
        
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            chars.append(self.vocab.get_token(tid))
            
        return "".join(chars)

    def save(self, filepath: str) -> None:
        """Saves tokenizer state."""
        self.vocab.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Loads tokenizer state."""
        vocab = Vocabulary.load(filepath)
        return cls(vocab=vocab)
