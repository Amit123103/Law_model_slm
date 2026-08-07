"""
End-to-End Automated SLM Pipeline Orchestrator.
Orchestrates downloading, cleaning, tokenization, dataset splitting, training,
evaluation, quality checking, and model card generation.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional

from slm.config.model_config import ModelConfig
from slm.config.train_config import TrainConfig
from slm.dataset.downloader import DatasetDownloader
from slm.dataset.cleaner import DataCleaner
from slm.dataset.manager import DatasetManager
from slm.dataset.prep import DatasetSplitter
from slm.dataset.dataset import CausalLMDataset
from slm.dataset.loader import create_dataloader
from slm.tokenizer.bpe import BPETokenizer
from slm.tokenizer.evaluator import TokenizerEvaluator
from slm.model.transformer_lm import SLMForCausalLM
from slm.training.trainer import Trainer
from slm.evaluation.evaluator import ModelEvaluator
from slm.utils.quality import QualityChecker
from slm.utils.logger import get_logger

logger = get_logger("slm.pipeline")


class PipelineOrchestrator:
    """Automated end-to-end pipeline runner for Small Language Models."""

    def __init__(self, output_dir: str = "checkpoints_pipeline"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.manager = DatasetManager(base_dir="data")

    def run_full_pipeline(
        self,
        raw_documents: List[str],
        vocab_size: int = 1000,
        max_steps: int = 200,
        epochs: int = 50,
        batch_size: int = 8,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Runs the entire 10-stage SLM lifecycle pipeline automatically."""

        start_time = time.time()
        logger.info("============================================================")
        logger.info("  STARTING AUTOMATED END-TO-END SLM PIPELINE")
        logger.info("============================================================")

        # STAGE 1 & 2: Cleaning & Preprocessing
        logger.info("[Stage 1/10] Cleaning and Preprocessing Corpus...")
        cleaner = DataCleaner(min_doc_chars=10, min_doc_words=2)
        cleaned_docs, clean_stats = cleaner.process_corpus(raw_documents)

        # STAGE 3: Tokenizer Training
        logger.info("[Stage 2/10] Training BPE Tokenizer from scratch...")
        tokenizer = BPETokenizer()
        tokenizer.train_on_texts(cleaned_docs, vocab_size=vocab_size)
        tok_dir = os.path.join(self.output_dir, "tokenizer")
        tokenizer.save(tok_dir)

        # STAGE 4: Tokenizer Evaluation
        logger.info("[Stage 3/10] Evaluating Tokenizer Quality...")
        tok_evaluator = TokenizerEvaluator(tokenizer)
        tok_report = tok_evaluator.full_evaluation(cleaned_docs[:10])

        # STAGE 5: Dataset Splitting & Preparation
        logger.info("[Stage 4/10] Splitting Dataset into Train / Val / Test...")
        splitter = DatasetSplitter(seed=42)
        train_docs, val_docs, test_docs = splitter.split_corpus(cleaned_docs, 0.8, 0.1, 0.1)
        splitter.save_splits(train_docs, val_docs, test_docs, output_dir=os.path.join("data", "splits"))

        # STAGE 6: Model Configuration & Instantiation
        logger.info("[Stage 5/10] Building Transformer LM Model...")
        model_config = ModelConfig(
            vocab_size=len(tokenizer.vocab),
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_seq_len=256,
            norm_type="rmsnorm",
            activation="swiglu",
            pos_encoding_type="rope"
        )
        model = SLMForCausalLM(model_config)

        # STAGE 7: Quality Pre-Flight Checks
        logger.info("[Stage 6/10] Running Pre-Flight Quality Checks...")
        QualityChecker.check_model_weights(model)
        QualityChecker.verify_tokenizer_consistency(tokenizer)

        # STAGE 8: DataLoaders & Training Loop
        logger.info("[Stage 7/10] Preparing DataLoaders and Launching Trainer...")
        train_dataset = CausalLMDataset(train_docs, tokenizer, max_seq_len=model_config.max_seq_len)
        val_dataset = CausalLMDataset(val_docs, tokenizer, max_seq_len=model_config.max_seq_len)

        train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

        train_config = TrainConfig(
            batch_size=batch_size,
            learning_rate=1.5e-3,
            max_steps=max_steps,
            epochs=epochs,
            device=device,
            output_dir=self.output_dir,
            save_interval_steps=50,
            log_interval_steps=50
        )

        trainer = Trainer(model, train_config, train_loader, val_loader=val_loader, tokenizer=tokenizer)
        trainer.train()

        # STAGE 9: Model Evaluation & Benchmarking
        logger.info("[Stage 8/10] Running Full Model Evaluation & Benchmarking...")
        evaluator = ModelEvaluator(model, tokenizer, device=device)
        eval_report = evaluator.full_evaluation_report(val_loader)

        # STAGE 10: Model Card & Final Summary Generation
        logger.info("[Stage 9/10] Generating Model Card and Summary Artifacts...")
        total_time = round(time.time() - start_time, 2)

        summary = {
            "pipeline_status": "SUCCESS",
            "total_execution_time_seconds": total_time,
            "cleaning_stats": clean_stats,
            "tokenizer_report": tok_report,
            "model_config": model_config.to_dict(),
            "evaluation_report": eval_report,
            "output_directory": self.output_dir
        }

        report_path = os.path.join(self.output_dir, "pipeline_summary.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("[Stage 10/10] Pipeline Complete! All artifacts saved.")
        return summary
