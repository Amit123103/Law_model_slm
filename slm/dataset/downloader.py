"""
Automated Dataset Downloader for Small Language Model training data.
Supports HTTP/HTTPS downloads, resumable transfers, checksum validation,
and automatic extraction of compressed archives (.gz, .zip, .tar.gz, .bz2).
"""

import os
import sys
import hashlib
import gzip
import zipfile
import tarfile
import bz2
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any
from slm.utils.logger import get_logger

logger = get_logger("slm.dataset.downloader")

# Pre-defined public dataset URLs for General, Legal, Instruction, and QA corpora
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gutenberg_sample": {
        "url": "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt",
        "format": "txt",
        "description": "Public domain English text corpus sample"
    },
    "wikitext2_sample": {
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "format": "txt",
        "description": "WikiText-2 Language Modeling Dataset"
    },
    "indian_constitution_sample": {
        "url": "https://raw.githubusercontent.com/law-dataset/indian-legal-sample/main/constitution.txt",
        "format": "txt",
        "description": "Indian Constitutional and Statute Legal Corpus Sample"
    },
    "alpaca_instruction_sample": {
        "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "format": "json",
        "description": "Stanford Alpaca Instruction Dataset Sample"
    }
}


class DatasetDownloader:
    """Handles downloading, checksum verification, and extraction of remote datasets."""

    def __init__(self, download_dir: str = "data/raw"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Computes the SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def download(
        self,
        url: str,
        filename: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        force_download: bool = False
    ) -> str:
        """
        Downloads a file from a URL with progress logging and optional SHA-256 verification.
        
        Args:
            url: The HTTP/HTTPS URL of the dataset file.
            filename: Target file name. If None, derived from the URL path.
            expected_sha256: Expected SHA-256 checksum for verification.
            force_download: Re-download even if local file exists and matches checksum.

        Returns:
            Absolute path to the downloaded file.
        """
        if not filename:
            filename = os.path.basename(urllib.parse.urlparse(url).path) or "dataset_download.bin"

        target_path = os.path.abspath(os.path.join(self.download_dir, filename))

        if os.path.exists(target_path) and not force_download:
            if expected_sha256:
                actual_sha = self.compute_sha256(target_path)
                if actual_sha.lower() == expected_sha256.lower():
                    logger.info(f"File already exists and passed SHA-256 verification: {target_path}")
                    return target_path
                else:
                    logger.warning(f"File exists but SHA-256 mismatch (Expected: {expected_sha256}, Got: {actual_sha}). Re-downloading...")
            else:
                logger.info(f"File already exists: {target_path}. Skipping download.")
                return target_path

        logger.info(f"Downloading dataset from {url} to {target_path}...")
        
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LawSLM-DatasetDownloader/1.0"}
        )
        
        with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
            chunk_size = 65536
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)

        logger.info(f"Successfully downloaded: {target_path}")

        if expected_sha256:
            actual_sha = self.compute_sha256(target_path)
            if actual_sha.lower() != expected_sha256.lower():
                raise ValueError(f"Checksum verification failed for {target_path}! Expected: {expected_sha256}, Got: {actual_sha}")

        return target_path

    def extract(self, archive_path: str, extract_to: Optional[str] = None) -> str:
        """
        Extracts compressed archives (.zip, .tar.gz, .tgz, .gz, .bz2) into a target directory.
        """
        if not extract_to:
            extract_to = self.download_dir

        os.makedirs(extract_to, exist_ok=True)
        logger.info(f"Extracting archive {archive_path} to {extract_to}...")

        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz") or archive_path.endswith(".tar"):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.endswith(".gz"):
            dest_file = os.path.join(extract_to, os.path.basename(archive_path)[:-3])
            with gzip.open(archive_path, "rb") as f_in, open(dest_file, "wb") as f_out:
                f_out.write(f_in.read())
            return dest_file
        elif archive_path.endswith(".bz2"):
            dest_file = os.path.join(extract_to, os.path.basename(archive_path)[:-4])
            with bz2.open(archive_path, "rb") as f_in, open(dest_file, "wb") as f_out:
                f_out.write(f_in.read())
            return dest_file
        else:
            logger.info(f"File {archive_path} is not a supported archive. Skipping extraction.")
            return archive_path

        logger.info(f"Extraction completed: {extract_to}")
        return extract_to

    def download_registered(self, dataset_name: str) -> str:
        """Downloads a dataset by name from the pre-defined DATASET_REGISTRY."""
        if dataset_name not in DATASET_REGISTRY:
            raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")
        
        meta = DATASET_REGISTRY[dataset_name]
        return self.download(url=meta["url"], filename=f"{dataset_name}.{meta['format']}")
