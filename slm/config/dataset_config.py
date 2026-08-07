"""
Dataset mixture configuration defining domain sources, URLs, ratios, and caching rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import yaml


@dataclass
class DatasetSourceConfig:
    name: str
    category: str  # "general", "instruction", "legal", "qa", "code"
    url: Optional[str] = None
    local_path: Optional[str] = None
    checksum: Optional[str] = None
    ratio: float = 1.0


@dataclass
class DatasetMixConfig:
    sources: List[DatasetSourceConfig] = field(default_factory=list)
    ratios: Dict[str, float] = field(default_factory=lambda: {
        "general": 0.40,
        "instruction": 0.25,
        "legal": 0.20,
        "qa": 0.10,
        "code": 0.05
    })
    cache_dir: str = "data/cache"
    output_dir: str = "data/mixed"

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMixConfig":
        sources = [DatasetSourceConfig(**src) for src in data.get("sources", [])]
        ratios = data.get("ratios", {
            "general": 0.40,
            "instruction": 0.25,
            "legal": 0.20,
            "qa": 0.10,
            "code": 0.05
        })
        return cls(
            sources=sources,
            ratios=ratios,
            cache_dir=data.get("cache_dir", "data/cache"),
            output_dir=data.get("output_dir", "data/mixed")
        )

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetMixConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict:
        return {
            "sources": [s.__dict__ for s in self.sources],
            "ratios": self.ratios,
            "cache_dir": self.cache_dir,
            "output_dir": self.output_dir
        }
