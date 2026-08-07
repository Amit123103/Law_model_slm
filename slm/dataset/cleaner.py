"""
Automated Data Cleaner and Preprocessor for SLM training datasets.
Handles Unicode NFKC normalization, HTML tag stripping, control character removal,
paragraph deduplication, minimum length filtering, and cleaning statistics generation.
"""

import re
import unicodedata
import hashlib
from typing import List, Dict, Any, Tuple
from slm.utils.logger import get_logger

logger = get_logger("slm.dataset.cleaner")


class DataCleaner:
    """Cleans, normalizes, deduplicates, and filters raw text corpora."""

    def __init__(self, min_doc_chars: int = 20, min_doc_words: int = 5):
        self.min_doc_chars = min_doc_chars
        self.min_doc_words = min_doc_words

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Applies Unicode NFKC normalization."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def strip_html_tags(text: str) -> str:
        """Removes HTML tags and unescapes HTML entities."""
        clean_re = re.compile(r"<[^>]+>")
        return clean_re.sub("", text)

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Removes non-printable control characters except for space and newline."""
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " "))

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalizes multiple horizontal spaces and trims blank lines."""
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Remove consecutive empty lines
        cleaned_lines: List[str] = []
        for line in lines:
            if line or (cleaned_lines and cleaned_lines[-1]):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def clean_document(self, text: str) -> str:
        """Applies full cleaning pipeline to a single document."""
        text = self.normalize_unicode(text)
        text = self.strip_html_tags(text)
        text = self.remove_control_characters(text)
        text = self.normalize_whitespace(text)
        return text

    def deduplicate_paragraphs(self, text: str) -> str:
        """Deduplicates identical paragraphs within a document."""
        paragraphs = text.split("\n\n")
        seen_hashes = set()
        unique_paragraphs = []

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            h = hashlib.md5(p_clean.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_paragraphs.append(p_clean)

        return "\n\n".join(unique_paragraphs)

    def is_valid_document(self, text: str) -> bool:
        """Checks if document meets minimum length and word count criteria."""
        if len(text) < self.min_doc_chars:
            return False
        words = text.split()
        if len(words) < self.min_doc_words:
            return False
        return True

    def process_corpus(self, raw_documents: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Cleans, deduplicates, and filters a list of raw documents while recording statistics.
        """
        cleaned_docs = []
        total_raw_bytes = 0
        total_cleaned_bytes = 0
        filtered_short_count = 0
        deduped_paragraphs_count = 0

        for doc in raw_documents:
            raw_bytes = len(doc.encode("utf-8"))
            total_raw_bytes += raw_bytes

            cleaned = self.clean_document(doc)
            deduped = self.deduplicate_paragraphs(cleaned)

            if len(cleaned) != len(deduped):
                deduped_paragraphs_count += 1

            if not self.is_valid_document(deduped):
                filtered_short_count += 1
                continue

            cleaned_bytes = len(deduped.encode("utf-8"))
            total_cleaned_bytes += cleaned_bytes
            cleaned_docs.append(deduped)

        stats = {
            "total_input_documents": len(raw_documents),
            "retained_documents": len(cleaned_docs),
            "filtered_documents": filtered_short_count,
            "raw_bytes": total_raw_bytes,
            "cleaned_bytes": total_cleaned_bytes,
            "compression_percentage": round((1.0 - (total_cleaned_bytes / (total_raw_bytes + 1e-9))) * 100, 2),
            "deduplicated_paragraphs_count": deduped_paragraphs_count
        }

        logger.info(f"Processed corpus: {stats['retained_documents']}/{stats['total_input_documents']} documents retained.")
        return cleaned_docs, stats
