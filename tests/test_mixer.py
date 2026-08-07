"""
Unit tests for slm.dataset.mixer.DatasetMixer.
"""

import pytest
from slm.dataset.mixer import DatasetMixer
from slm.config.dataset_config import DatasetMixConfig


def test_dataset_mixer_ratios():
    config = DatasetMixConfig.from_dict({
        "ratios": {
            "general": 0.50,
            "legal": 0.30,
            "instruction": 0.20
        }
    })

    mixer = DatasetMixer(config)
    corpora = {
        "general": [f"General doc {i}" for i in range(100)],
        "legal": [f"Legal section {i}" for i in range(50)],
        "instruction": [f"User: hi\nSLM: Hello {i}" for i in range(40)]
    }

    mixed = mixer.mix_corpora(corpora, target_total_docs=100, seed=42)
    assert len(mixed) == 100
    assert any("General doc" in d for d in mixed)
    assert any("Legal section" in d for d in mixed)
    assert any("User: hi" in d for d in mixed)


def test_dataset_mixer_fallback():
    mixer = DatasetMixer()
    corpora = {
        "unknown_domain": ["Doc A", "Doc B", "Doc C"]
    }
    mixed = mixer.mix_corpora(corpora)
    assert len(mixed) == 3
