"""
Performance and hardware benchmarking suite for SLM.
"""

import time
from typing import Dict, Any, List
import torch
import torch.nn as nn

from slm.config.model_config import ModelConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.utils.logger import get_logger
from slm.utils.utils import get_memory_stats

logger = get_logger("slm.benchmark")


def benchmark_inference(
    model: SLMForCausalLM,
    batch_size: int = 1,
    seq_len: int = 256,
    num_runs: int = 50,
    warmup_runs: int = 10,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Benchmarks forward pass inference throughput and latency.

    Returns:
        Dictionary containing average latency (ms), throughput (tokens/sec), and VRAM usage.
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
    )

    # Warmup iterations
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000.0
    total_tokens = batch_size * seq_len * num_runs
    tokens_per_sec = total_tokens / total_time

    mem_stats = get_memory_stats(device)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_tokens_benchmarked": total_tokens,
        "device": str(device),
        "memory_stats": mem_stats
    }


def benchmark_training_step(
    model: SLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 8,
    seq_len: int = 256,
    num_runs: int = 20,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Benchmarks training iteration step (forward + backward + optimizer step).
    """
    model.train()
    model.to(device)

    dummy_input = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
    )
    dummy_targets = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
    )

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        _, loss = model(dummy_input, targets=dummy_targets)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_runs):
        optimizer.zero_grad()
        _, loss = model(dummy_input, targets=dummy_targets)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_step_ms = (total_time / num_runs) * 1000.0

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_step_ms": round(avg_step_ms, 2),
        "steps_per_sec": round(num_runs / total_time, 2),
        "device": str(device)
    }
