"""
Unit tests for DatasetDownloader and DatasetManager.
"""

import os
import tempfile
import pytest
from slm.dataset.downloader import DatasetDownloader
from slm.dataset.manager import DatasetManager


def test_dataset_downloader_download_and_checksum():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = DatasetDownloader(download_dir=tmpdir)
        
        # Test registered dataset download (WikiText-2 sample)
        path = downloader.download_registered("wikitext2_sample")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

        # Test checksum calculation
        sha256 = downloader.compute_sha256(path)
        assert len(sha256) == 64


def test_dataset_manager_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DatasetManager(base_dir=tmpdir, raw_dir=f"{tmpdir}/raw", processed_dir=f"{tmpdir}/processed", splits_dir=f"{tmpdir}/splits")
        manager.log_download("test_dataset", "http://example.com/data.txt", f"{tmpdir}/raw/data.txt", 1024)
        
        history = manager.get_download_history()
        assert len(history) == 1
        assert history[0]["dataset_name"] == "test_dataset"
        assert history[0]["file_size_bytes"] == 1024
