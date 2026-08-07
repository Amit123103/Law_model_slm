"""
Script to preprocess, clean, deduplicate, and format raw text datasets.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slm.dataset.cleaner import TextCleaner
from slm.dataset.readers import DatasetReader
from slm.utils.logger import get_logger

logger = get_logger("slm.scripts.preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and clean raw text files.")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory path")
    parser.add_argument("--output", type=str, required=True, help="Output destination file path")
    args = parser.parse_args()

    logger.info(f"Loading raw data from {args.input}...")
    if os.path.isdir(args.input):
        docs = DatasetReader.load_directory(args.input)
    else:
        docs = DatasetReader.load_file(args.input)

    logger.info(f"Loaded {len(docs)} raw documents. Cleaning and deduplicating...")
    cleaner = TextCleaner()
    cleaned_docs = [cleaner.clean_text(d) for d in docs]
    deduped_docs = cleaner.deduplicate(cleaned_docs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in deduped_docs:
            f.write(doc + "\n")

    logger.info(f"Preprocessed corpus saved to {args.output} ({len(deduped_docs)} unique clean documents).")


if __name__ == "__main__":
    main()
