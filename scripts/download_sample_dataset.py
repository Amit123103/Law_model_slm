"""
Dataset Downloader Utility for fetching open-source text corpora for training Small Language Models.
"""

import argparse
import os
import urllib.request
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slm.utils.logger import get_logger

logger = get_logger("slm.scripts.download_dataset")

DATASET_URLS = {
    "tinystories": {
        "url": "https://raw.githubusercontent.com/roneneldan/TinyStories/main/TinyStories-valid.txt",
        "filename": "tinystories_valid.txt",
        "description": "Clean synthetic story corpus ideal for small language model training."
    },
    "wikitext2": {
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "filename": "wikitext2_train.txt",
        "description": "High-quality Wikipedia text corpus (WikiText-2)."
    },
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "tinyshakespeare.txt",
        "description": "Classic Shakespeare text corpus (1MB)."
    }
}


def download_dataset(dataset_key: str, output_dir: str = "data") -> str:
    """
    Downloads specified open text dataset into target directory.
    """
    if dataset_key not in DATASET_URLS:
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Choose from: {list(DATASET_URLS.keys())}")

    info = DATASET_URLS[dataset_key]
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, info["filename"])

    logger.info(f"Downloading {dataset_key} ({info['description']})...")
    logger.info(f"Source URL: {info['url']}")

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / max(1, total_size))
        sys.stdout.write(f"\rDownloading... {percent}% complete")
        sys.stdout.flush()

    urllib.request.urlretrieve(info["url"], dest_path, reporthook=_progress)
    print()
    logger.info(f"Successfully downloaded dataset to {dest_path}")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Download open text datasets for Small Language Model training.")
    parser.add_argument("--name", type=str, default="wikitext2", choices=list(DATASET_URLS.keys()), help="Dataset choice")
    parser.add_argument("--output_dir", type=str, default="data", help="Output folder")
    args = parser.parse_args()

    download_dataset(args.name, args.output_dir)


if __name__ == "__main__":
    main()
