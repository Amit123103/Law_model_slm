"""
Unit tests for DataCleaner.
"""

import pytest
from slm.dataset.cleaner import DataCleaner


def test_data_cleaner_functions():
    cleaner = DataCleaner(min_doc_chars=10, min_doc_words=2)

    # 1. HTML stripping & Unicode normalization
    dirty_text = "<p>Hello &amp; World!\u00A0</p>\x00"
    cleaned = cleaner.clean_document(dirty_text)
    assert "<p>" not in cleaned
    assert "Hello" in cleaned
    assert "World!" in cleaned

    # 2. Paragraph deduplication
    dup_text = "Paragraph One.\n\nParagraph Two.\n\nParagraph One."
    deduped = cleaner.deduplicate_paragraphs(dup_text)
    assert deduped == "Paragraph One.\n\nParagraph Two."

    # 3. Document validation
    assert cleaner.is_valid_document("This is a valid test document.")
    assert not cleaner.is_valid_document("Short")


def test_process_corpus():
    cleaner = DataCleaner(min_doc_chars=10, min_doc_words=2)
    corpus = [
        "<h1>Title</h1>\n<p>This is a valid long document for testing.</p>",
        "Too short",
        "Duplicate paragraph.\n\nDuplicate paragraph."
    ]
    retained, stats = cleaner.process_corpus(corpus)
    assert stats["total_input_documents"] == 3
    assert stats["retained_documents"] == 2
    assert stats["filtered_documents"] == 1
