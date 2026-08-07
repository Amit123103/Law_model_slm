"""
Evaluation and Benchmarking Engine for Small Language Model validation.
Computes validation loss, perplexity, top-1/top-5 next-token prediction accuracy,
inference latency (ms/token, tokens/sec), and memory footprint metrics.
"""

import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from slm.model.transformer_lm import SLMForCausalLM
from slm.tokenizer.bpe import BPETokenizer
from slm.sampling.generator import TextGenerator
from slm.utils.logger import get_logger

logger = get_logger("slm.evaluation")


class ModelEvaluator:
    """Evaluates Causal LM performance, perplexity, accuracy, and inference latency."""

    def __init__(self, model: SLMForCausalLM, tokenizer: BPETokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate_loss_and_perplexity(self, dataloader) -> Dict[str, float]:
        """Calculates average cross-entropy loss and perplexity across a validation dataloader."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    input_ids, target_ids = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    target_ids = batch["target_ids"].to(self.device)

                logits, loss = self.model(input_ids, targets=target_ids)
                if not math.isnan(loss.item()) and not math.isinf(loss.item()):
                    total_loss += loss.item()
                    total_batches += 1

        avg_loss = total_loss / (total_batches + 1e-9)
        perplexity = math.exp(avg_loss) if avg_loss < 100.0 else float("inf")

        return {
            "val_loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 4)
        }

    def evaluate_topk_accuracy(self, dataloader, k_list: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
        """Computes top-1 and top-5 next-token prediction accuracy."""
        self.model.eval()
        correct_counts = {k: 0 for k in k_list}
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    input_ids, target_ids = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    target_ids = batch["target_ids"].to(self.device)

                logits, _ = self.model(input_ids)  # [B, T, V]
                # Flatten
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = target_ids.view(-1)

                valid_mask = targets_flat != -100
                logits_valid = logits_flat[valid_mask]
                targets_valid = targets_flat[valid_mask]

                if targets_valid.numel() == 0:
                    continue

                for k in k_list:
                    _, topk_indices = torch.topk(logits_valid, k=k, dim=-1)
                    targets_expanded = targets_valid.unsqueeze(1).expand_as(topk_indices)
                    matches = (topk_indices == targets_expanded).any(dim=1).sum().item()
                    correct_counts[k] += matches

                total_tokens += targets_valid.numel()

        results = {}
        for k in k_list:
            acc = (correct_counts[k] / (total_tokens + 1e-9)) * 100.0
            results[f"top_{k}_accuracy"] = round(acc, 2)

        return results

    def measure_inference_latency(
        self,
        prompt: str = "LawSLM is a small language model.",
        max_new_tokens: int = 50,
        warmup_runs: int = 1,
        benchmark_runs: int = 3
    ) -> Dict[str, float]:
        """Measures token generation speed (tokens/sec and ms/token)."""
        generator = TextGenerator(self.model, self.tokenizer)

        # Warmup
        for _ in range(warmup_runs):
            _ = generator.generate(prompt, max_new_tokens=10, temperature=0.7)

        durations = []
        tokens_generated_list = []

        for _ in range(benchmark_runs):
            start = time.perf_counter()
            output = generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.7)
            end = time.perf_counter()

            dur = end - start
            gen_tokens = max(1, len(self.tokenizer.encode(output)) - len(self.tokenizer.encode(prompt)))

            durations.append(dur)
            tokens_generated_list.append(gen_tokens)

        avg_duration = sum(durations) / len(durations)
        total_gen_tokens = sum(tokens_generated_list)
        tokens_per_sec = total_gen_tokens / sum(durations)
        ms_per_token = (avg_duration / (total_gen_tokens / len(durations))) * 1000.0

        return {
            "avg_latency_seconds": round(avg_duration, 4),
            "tokens_per_second": round(tokens_per_sec, 2),
            "ms_per_token": round(ms_per_token, 2)
        }

    def full_evaluation_report(self, dataloader) -> Dict[str, Any]:
        """Runs full evaluation suite and returns consolidated report dictionary."""
        loss_ppl = self.evaluate_loss_and_perplexity(dataloader)
        topk_acc = self.evaluate_topk_accuracy(dataloader)
        latency = self.measure_inference_latency()

        report = {
            "val_loss": loss_ppl["val_loss"],
            "perplexity": loss_ppl["perplexity"],
            "top_1_accuracy": topk_acc["top_1_accuracy"],
            "top_5_accuracy": topk_acc["top_5_accuracy"],
            "tokens_per_second": latency["tokens_per_second"],
            "ms_per_token": latency["ms_per_token"]
        }

        logger.info(
            f"Evaluation Report -> Loss: {report['val_loss']}, PPL: {report['perplexity']}, "
            f"Top-1 Acc: {report['top_1_accuracy']}%, Speed: {report['tokens_per_second']} tok/s"
        )
        return report
