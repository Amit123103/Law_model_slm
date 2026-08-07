"""
Tokenizer Quality Evaluator for assessing BPE tokenization performance,
encode/decode roundtrip fidelity, compression ratios, and UNK token rates.
"""

from typing import List, Dict, Any
from slm.tokenizer.bpe import BPETokenizer
from slm.utils.logger import get_logger

logger = get_logger("slm.tokenizer.evaluator")


class TokenizerEvaluator:
    """Evaluates tokenizer quality, compression efficiency, and reconstruction fidelity."""

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer

    def evaluate_roundtrip(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Verifies that encode(decode(text)) reproduces the original text accurately.
        """
        exact_matches = 0
        total_samples = len(sample_texts)

        for text in sample_texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=True)
            if decoded.strip() == text.strip():
                exact_matches += 1

        accuracy = (exact_matches / total_samples) * 100 if total_samples > 0 else 0.0
        return {
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "roundtrip_accuracy": round(accuracy, 2)
        }

    def evaluate_compression(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Calculates character-to-token ratio and byte-to-token compression ratio.
        """
        total_bytes = 0
        total_chars = 0
        total_tokens = 0
        unk_count = 0

        unk_id = self.tokenizer.vocab.unk_id

        for text in sample_texts:
            total_bytes += len(text.encode("utf-8"))
            total_chars += len(text)
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(encoded)
            unk_count += encoded.count(unk_id)

        bytes_per_token = total_bytes / (total_tokens + 1e-9)
        chars_per_token = total_chars / (total_tokens + 1e-9)
        unk_percentage = (unk_count / (total_tokens + 1e-9)) * 100

        return {
            "total_bytes": total_bytes,
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "bytes_per_token": round(bytes_per_token, 2),
            "chars_per_token": round(chars_per_token, 2),
            "unk_count": unk_count,
            "unk_percentage": round(unk_percentage, 4)
        }

    def full_evaluation(self, sample_texts: List[str]) -> Dict[str, Any]:
        """Runs full suite of tokenizer quality metrics."""
        roundtrip = self.evaluate_roundtrip(sample_texts)
        compression = self.evaluate_compression(sample_texts)
        vocab_size = len(self.tokenizer.vocab)

        report = {
            "vocab_size": vocab_size,
            "roundtrip": roundtrip,
            "compression": compression
        }

        logger.info(
            f"Tokenizer Evaluation: Vocab={vocab_size}, "
            f"Roundtrip={roundtrip['roundtrip_accuracy']}%, "
            f"Bytes/Token={compression['bytes_per_token']}, "
            f"UNK Rate={compression['unk_percentage']}%"
        )
        return report
