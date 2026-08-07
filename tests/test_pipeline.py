"""
Unit tests for PipelineOrchestrator end-to-end pipeline execution.
"""

import os
import tempfile
import pytest
from slm.pipeline import PipelineOrchestrator


def test_pipeline_orchestrator():
    corpus = [
        "User: hello\nSLM: Hello! How can I assist you with language modeling today?\n",
        "User: What is law?\nSLM: Law is a system of rules created and enforced by institutions.\n",
        "User: What is Python?\nSLM: Python is a high-level programming language.\n"
    ] * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = PipelineOrchestrator(output_dir=tmpdir)
        summary = orchestrator.run_full_pipeline(
            raw_documents=corpus,
            vocab_size=150,
            max_steps=20,
            epochs=5,
            batch_size=2,
            device="cpu"
        )

        assert summary["pipeline_status"] == "SUCCESS"
        assert summary["total_execution_time_seconds"] > 0
        assert os.path.isfile(os.path.join(tmpdir, "pipeline_summary.json"))
        assert os.path.isdir(os.path.join(tmpdir, "tokenizer"))
