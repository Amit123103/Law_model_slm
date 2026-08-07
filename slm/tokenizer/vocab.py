"""
Vocabulary manager and dictionary statistics helper.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Set, Any


class Vocabulary:
    """
    Vocabulary container providing bidirection token <-> integer id mappings,
    special token management, and JSON serialization.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    MASK_TOKEN = "<mask>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        """
        Initializes vocabulary data structures with special tokens.

        Args:
            special_tokens: Additional custom special token strings.
        """
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.frequencies: Dict[str, int] = {}

        # Add default special tokens
        tokens_to_add = list(self.SPECIAL_TOKENS)
        if special_tokens:
            for st in special_tokens:
                if st not in tokens_to_add:
                    tokens_to_add.append(st)

        for token in tokens_to_add:
            self.add_token(token, is_special=True)

    def add_token(self, token: str, is_special: bool = False, count: int = 1) -> int:
        """
        Adds a token to the vocabulary if absent and updates frequency count.

        Args:
            token: String representation of token.
            is_special: Flag indicating if token is special.
            count: Frequency count increment.

        Returns:
            Assigned integer ID of the token.
        """
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
            self.frequencies[token] = count
        else:
            self.frequencies[token] += count
            token_id = self.token2id[token]
        return token_id

    def __len__(self) -> int:
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        return token in self.token2id

    def get_id(self, token: str) -> int:
        """Returns integer ID for token, defaulting to UNK_TOKEN ID if missing."""
        return self.token2id.get(token, self.token2id.get(self.UNK_TOKEN, 1))

    def get_token(self, token_id: int) -> str:
        """Returns string token for integer ID, defaulting to UNK_TOKEN if missing."""
        return self.id2token.get(token_id, self.UNK_TOKEN)

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token2id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token2id[self.EOS_TOKEN]

    @property
    def mask_id(self) -> int:
        return self.token2id[self.MASK_TOKEN]

    def prune(self, min_freq: int = 1, max_vocab_size: Optional[int] = None) -> None:
        """
        Prunes non-special tokens from vocabulary based on minimum frequency and max vocab capacity.

        Args:
            min_freq: Minimum token occurrence frequency required.
            max_vocab_size: Upper threshold for vocabulary size.
        """
        specials = set(self.SPECIAL_TOKENS)
        kept_tokens = [t for t in self.SPECIAL_TOKENS]

        regular_tokens = [
            (t, freq) for t, freq in self.frequencies.items()
            if t not in specials and freq >= min_freq
        ]
        regular_tokens.sort(key=lambda x: x[1], reverse=True)

        if max_vocab_size is not None:
            max_regular = max(0, max_vocab_size - len(kept_tokens))
            regular_tokens = regular_tokens[:max_regular]

        kept_tokens.extend([t for t, _ in regular_tokens])

        # Re-index vocabulary
        self.token2id = {}
        self.id2token = {}
        new_freqs = {}
        for idx, token in enumerate(kept_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token
            new_freqs[token] = self.frequencies.get(token, 1)

        self.frequencies = new_freqs

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculates vocabulary statistical summary.

        Returns:
            Dictionary containing size, special tokens, and frequency ranges.
        """
        return {
            "vocab_size": len(self),
            "num_special_tokens": len(self.SPECIAL_TOKENS),
            "max_frequency": max(self.frequencies.values()) if self.frequencies else 0,
            "min_frequency": min(self.frequencies.values()) if self.frequencies else 0,
        }

    def save(self, filepath: str) -> None:
        """Saves vocabulary state to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "token2id": self.token2id,
            "frequencies": self.frequencies,
            "special_tokens": self.SPECIAL_TOKENS
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Loads vocabulary state from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(special_tokens=data.get("special_tokens"))
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(v): k for k, v in data["token2id"].items()}
        vocab.frequencies = data.get("frequencies", {})
        return vocab
