"""
Autoregressive Text Generator supporting Temperature, Top-K, Top-P, Penalties, and Streaming.
"""

from typing import List, Optional, Callable, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.model.transformer_lm import SLMForCausalLM
from slm.tokenizer.bpe import BPETokenizer
from slm.utils.logger import get_logger

logger = get_logger("slm.sampling")


class TextGenerator:
    """
    Inference and sampling engine for SLMForCausalLM.
    Supports Greedy, Temperature, Top-K, Top-P (Nucleus), Typical sampling,
    Repetition/Frequency/Presence penalties, and streaming token callbacks.
    """

    def __init__(self, model: SLMForCausalLM, tokenizer: BPETokenizer) -> None:
        """
        Initializes TextGenerator.

        Args:
            model: Trained SLMForCausalLM model instance.
            tokenizer: BPETokenizer instance for encoding prompt and decoding tokens.
        """
        self.model = model
        self.tokenizer = tokenizer

    def _apply_penalties(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> torch.Tensor:
        """
        Applies repetition, frequency, and presence penalty adjustments to raw token logits.
        """
        if not generated_tokens:
            return logits

        token_counts: Dict[int, int] = {}
        for token_id in generated_tokens:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

        for token_id, count in token_counts.items():
            if token_id >= logits.size(-1):
                continue
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

            # Frequency penalty
            if frequency_penalty > 0.0:
                logits[0, token_id] -= frequency_penalty * count

            # Presence penalty
            if presence_penalty > 0.0:
                logits[0, token_id] -= presence_penalty

        return logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        typical_p: float = 1.0,
        min_p: float = 0.0
    ) -> int:
        """
        Samples a single token ID from output logits using specified sampling strategy.
        """
        # logits shape: [1, vocab_size]
        logits = logits / max(temperature, 1e-5)

        # Exclude pad and unk special tokens from generation candidates
        pad_id = self.tokenizer.vocab.pad_id
        unk_id = self.tokenizer.vocab.unk_id
        logits[..., pad_id] = float("-inf")
        logits[..., unk_id] = float("-inf")

        # Minimum probability filtering
        if min_p > 0.0:
            probs = F.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            threshold = min_p * max_prob
            logits[probs < threshold] = float("-inf")

        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Compute final probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample or Greedy selection
        if temperature <= 1e-5:
            next_token_id = torch.argmax(probs, dim=-1).item()
        else:
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        return int(next_token_id)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_tokens: Optional[List[str]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        device: str = "auto"
    ) -> str:
        """
        Generates text continuation for prompt text string.

        Args:
            prompt: Text prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Softmax sampling temperature.
            top_k: Top-K filter limit.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Multiplicative penalty for repeated tokens.
            frequency_penalty: Additive penalty per token frequency.
            presence_penalty: Additive penalty for token presence.
            stop_tokens: List of string stop sequences.
            stream_callback: Optional callback receiving individual streamed tokens.
            device: Execution device.

        Returns:
            Complete generated output text.
        """
        self.model.eval()
        dev = next(self.model.parameters()).device

        # Encode prompt (prepend BOS token, do not append EOS to generation prompt)
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_token_ids or prompt_token_ids[0] != self.tokenizer.vocab.bos_id:
            prompt_token_ids = [self.tokenizer.vocab.bos_id] + prompt_token_ids

        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=dev)
        generated_ids: List[int] = list(prompt_token_ids)

        stop_token_ids = {self.tokenizer.vocab.eos_id}
        if stop_tokens:
            for st in stop_tokens:
                st_ids = self.tokenizer.encode(st, add_special_tokens=False)
                stop_token_ids.update(st_ids)

        max_ctx = self.model.config.max_seq_len

        for step in range(max_new_tokens):
            # Truncate context sequence if exceeding model context window
            curr_input = input_ids[:, -max_ctx:]

            logits, _ = self.model(curr_input)
            # Focus on logits of final position: [1, vocab_size]
            next_logits = logits[:, -1, :]

            # Apply repetition penalties
            next_logits = self._apply_penalties(
                logits=next_logits,
                generated_tokens=generated_ids,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )

            # Sample next token
            next_token = self._sample_next_token(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            generated_ids.append(next_token)
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=dev)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)

            # Streaming callback trigger
            if stream_callback is not None:
                token_str = self.tokenizer.decode([next_token], skip_special_tokens=True)
                stream_callback(token_str)

            # Check stop condition
            if next_token in stop_token_ids:
                break
                
            # Stop if generating new turn boundary ("User:" or "\nUser")
            recent_text = self.tokenizer.decode(generated_ids[len(prompt_token_ids):], skip_special_tokens=True)
            if "\nUser" in recent_text or "\nUser:" in recent_text or "User:" in recent_text:
                break

        full_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return full_output
