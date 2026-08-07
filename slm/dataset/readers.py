"""
Multi-format dataset ingestion system supporting TXT, CSV, JSON, JSONL, MD, HTML, XML.
"""

import csv
import json
import os
import re
from typing import List, Dict, Any, Optional
from slm.utils.logger import get_logger

logger = get_logger("slm.dataset.readers")


class DatasetReader:
    """
    Unified multi-format file reader extracting text corpora from various file formats.
    """

    @staticmethod
    def read_txt(filepath: str) -> List[str]:
        """Reads plain text file into lines or document strings."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [content] if content.strip() else []

    @staticmethod
    def read_csv(filepath: str, text_column: str = "text") -> List[str]:
        """Reads CSV file and extracts values from designated text column."""
        documents = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row and row[text_column]:
                    documents.append(row[text_column])
        return documents

    @staticmethod
    def read_json(filepath: str, text_key: str = "text") -> List[str]:
        """Reads JSON file containing list of objects or text array."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        if isinstance(data, list):
            docs = []
            for item in data:
                if isinstance(item, str):
                    docs.append(item)
                elif isinstance(item, dict) and text_key in item:
                    docs.append(str(item[text_key]))
            return docs
        elif isinstance(data, dict) and text_key in data:
            return [str(data[text_key])]
        return []

    @staticmethod
    def read_jsonl(filepath: str, text_key: str = "text") -> List[str]:
        """Reads line-delimited JSONL file."""
        documents = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line.strip())
                if isinstance(obj, dict) and text_key in obj:
                    documents.append(str(obj[text_key]))
                elif isinstance(obj, str):
                    documents.append(obj)
        return documents

    @staticmethod
    def read_markdown(filepath: str) -> List[str]:
        """Reads markdown file and strips code headers and formatting tags."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        cleaned = re.sub(r"```[\s\S]*?```", "", content)
        cleaned = re.sub(r"#+\s*", "", cleaned)
        return [cleaned] if cleaned.strip() else []

    @staticmethod
    def read_pdf_hook(filepath: str) -> List[str]:
        """
        Extensible hook for PDF text extraction.
        """
        logger.warning(f"PDF extraction hook triggered for {filepath}. Requires external pdf miner library.")
        return []

    @classmethod
    def load_file(cls, filepath: str, text_key: str = "text") -> List[str]:
        """
        Auto-detects file type by extension and delegates to appropriate parser method.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".txt":
            return cls.read_txt(filepath)
        elif ext == ".csv":
            return cls.read_csv(filepath, text_column=text_key)
        elif ext == ".json":
            return cls.read_json(filepath, text_key=text_key)
        elif ext == ".jsonl":
            return cls.read_jsonl(filepath, text_key=text_key)
        elif ext in (".md", ".markdown"):
            return cls.read_markdown(filepath)
        elif ext == ".pdf":
            return cls.read_pdf_hook(filepath)
        else:
            logger.warning(f"Unknown format '{ext}'. Fallback to text reading for {filepath}")
            return cls.read_txt(filepath)

    @classmethod
    def load_directory(cls, directory_path: str, text_key: str = "text") -> List[str]:
        """
        Recursively scans directory and ingests all supported document files.
        """
        all_documents = []
        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                full_path = os.path.join(root, file)
                docs = cls.load_file(full_path, text_key=text_key)
                all_documents.extend(docs)
        return all_documents
