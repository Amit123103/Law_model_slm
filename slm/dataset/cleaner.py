"""
Dataset preprocessing, text cleaning, normalization, deduplication, and sentence splitting.
"""

import re
import unicodedata
from typing import List, Set, Dict, Any, Optional


class TextCleaner:
    """
    Comprehensive text cleaning and normalization pipeline for raw text corpora.
    """

    def __init__(self, strip_html: bool = True, normalize_unicode: bool = True) -> None:
        self.strip_html = strip_html
        self.normalize_unicode = normalize_unicode

    def clean_text(self, text: str) -> str:
        """
        Applies cleaning operations to a string of raw text.

        Args:
            text: Input raw text string.

        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""

        # 1. Unicode normalization (NFC)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # 2. Strip HTML/XML tags
        if self.strip_html:
            text = re.sub(r"<[^>]+>", " ", text)

        # 3. Normalize whitespace (collapse multiple spaces/tabs into single space)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def deduplicate(texts: List[str]) -> List[str]:
        """
        Removes exact duplicate documents while preserving sequence order.

        Args:
            texts: List of text documents.

        Returns:
            Deduplicated list of text documents.
        """
        seen: Set[str] = set()
        unique_texts: List[str] = []
        for text in texts:
            cleaned = text.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                unique_texts.append(cleaned)
        return unique_texts

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Splits text into individual sentences using punctuation boundaries.
        """
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def chunk_document(text: str, max_words_per_chunk: int = 512, overlap: int = 64) -> List[str]:
        """
        Chunks text into overlapping fixed-size word windows.

        Args:
            text: Document text string.
            max_words_per_chunk: Maximum words per chunk.
            overlap: Overlapping word count between consecutive chunks.

        Returns:
            List of chunk strings.
        """
        words = text.strip().split()
        if len(words) <= max_words_per_chunk:
            return [text]

        chunks = []
        step = max_words_per_chunk - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + max_words_per_chunk]
            if len(chunk_words) > 0:
                chunks.append(" ".join(chunk_words))

        return chunks
