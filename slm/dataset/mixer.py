"""
DatasetMixer for multi-source dataset blending according to domain target ratios.
"""

import random
from typing import Dict, List, Optional
from slm.config.dataset_config import DatasetMixConfig
from slm.utils.logger import get_logger

logger = get_logger("slm.dataset.mixer")


class DatasetMixer:
    """
    Blends multi-domain datasets (General, Instruction, Legal, QA, Code) according to configurable mixture ratios.
    """

    def __init__(self, config: Optional[DatasetMixConfig] = None) -> None:
        """
        Initializes DatasetMixer with a DatasetMixConfig.
        """
        self.config = config or DatasetMixConfig()

    def mix_corpora(
        self,
        domain_corpora: Dict[str, List[str]],
        target_total_docs: Optional[int] = None,
        seed: int = 42
    ) -> List[str]:
        """
        Mixes documents from multiple domain corpora based on configured ratios.

        Args:
            domain_corpora: Dictionary mapping category/domain to document lists.
                            e.g. {"general": [...], "legal": [...], "instruction": [...]}
            target_total_docs: Total target document count (defaults to total available).
            seed: Random seed for deterministic shuffling.

        Returns:
            Shuffled list of blended document texts.
        """
        rng = random.Random(seed)
        ratios = self.config.ratios

        # Normalize ratios if categories exist
        available_categories = [cat for cat in ratios if cat in domain_corpora and domain_corpora[cat]]
        if not available_categories:
            # Fallback: flatten all available corpora
            logger.warning("No matching domain categories found in corpora map. Flattening all documents.")
            flat_docs: List[str] = []
            for docs in domain_corpora.values():
                flat_docs.extend(docs)
            rng.shuffle(flat_docs)
            return flat_docs

        total_ratio = sum(ratios[cat] for cat in available_categories)
        norm_ratios = {cat: ratios[cat] / total_ratio for cat in available_categories}

        # Calculate sample counts per category
        if target_total_docs is None:
            # Estimate based on largest available relative to ratio
            target_total_docs = sum(len(docs) for docs in domain_corpora.values())

        mixed_docs: List[str] = []
        for cat in available_categories:
            docs = domain_corpora[cat]
            if not docs:
                continue

            n_samples = max(1, int(round(target_total_docs * norm_ratios[cat])))
            if len(docs) >= n_samples:
                sampled = rng.sample(docs, n_samples)
            else:
                # Oversample with replacement if domain has fewer documents
                sampled = [rng.choice(docs) for _ in range(n_samples)]

            mixed_docs.extend(sampled)
            logger.info(f"Domain '{cat}': sampled {len(sampled)} documents (Ratio: {norm_ratios[cat]:.1%})")

        rng.shuffle(mixed_docs)
        logger.info(f"DatasetMixer complete: blended {len(mixed_docs)} total documents across {len(available_categories)} domains.")
        return mixed_docs
