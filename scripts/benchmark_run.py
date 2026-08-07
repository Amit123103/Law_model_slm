"""
Script to execute SLM inference and training benchmarks.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slm.config.model_config import ModelConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.evaluation.benchmark import benchmark_inference, benchmark_training_step
from slm.utils.logger import get_logger
from slm.utils.utils import get_device

logger = get_logger("slm.scripts.benchmark")


def run() -> None:
    device = get_device("auto")
    logger.info(f"Starting hardware benchmark on device={device}...")

    # Benchmark Nano model
    nano_cfg = ModelConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=2, d_ff=512)
    nano_model = SLMForCausalLM(nano_cfg)
    nano_inf = benchmark_inference(nano_model, batch_size=4, seq_len=128, device=device)

    logger.info("=== Nano Model Inference Benchmark ===")
    logger.info(f"Avg Latency: {nano_inf['avg_latency_ms']} ms")
    logger.info(f"Throughput:  {nano_inf['tokens_per_sec']} tokens/sec")

    # Benchmark Standard model
    std_cfg = ModelConfig(vocab_size=32000, d_model=512, n_heads=8, n_layers=8, d_ff=2048)
    std_model = SLMForCausalLM(std_cfg)
    std_inf = benchmark_inference(std_model, batch_size=2, seq_len=256, device=device)

    logger.info("=== Standard Model Inference Benchmark ===")
    logger.info(f"Avg Latency: {std_inf['avg_latency_ms']} ms")
    logger.info(f"Throughput:  {std_inf['tokens_per_sec']} tokens/sec")


if __name__ == "__main__":
    run()
