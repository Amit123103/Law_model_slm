"""
Dataset Manager for tracking versions, directory structure, logging, and dataset metadata.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from slm.utils.logger import get_logger

logger = get_logger("slm.dataset.manager")


class DatasetManager:
    """Manages raw, processed, and split dataset directories and tracks dataset versions."""

    def __init__(
        self,
        base_dir: str = "data",
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        splits_dir: str = "data/splits"
    ):
        self.base_dir = base_dir
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.splits_dir = splits_dir
        self.log_file = os.path.join(self.base_dir, "download_log.json")

        for d in [self.base_dir, self.raw_dir, self.processed_dir, self.splits_dir]:
            os.makedirs(d, exist_ok=True)

    def log_download(self, dataset_name: str, source_url: str, filepath: str, file_size_bytes: int) -> None:
        """Logs a dataset download event to the download history JSON file."""
        history = self.get_download_history()
        entry = {
            "dataset_name": dataset_name,
            "source_url": source_url,
            "filepath": filepath,
            "file_size_bytes": file_size_bytes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        history.append(entry)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Logged dataset download: {dataset_name}")

    def get_download_history(self) -> List[Dict[str, Any]]:
        """Reads and returns the download history log."""
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def list_datasets(self) -> Dict[str, List[str]]:
        """Lists available raw, processed, and split dataset files."""
        return {
            "raw": os.listdir(self.raw_dir) if os.path.exists(self.raw_dir) else [],
            "processed": os.listdir(self.processed_dir) if os.path.exists(self.processed_dir) else [],
            "splits": os.listdir(self.splits_dir) if os.path.exists(self.splits_dir) else [],
        }
