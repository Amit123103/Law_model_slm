"""
Interactive REPL Chat script for asking questions and chatting with the Small Language Model.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from slm.config.model_config import ModelConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.tokenizer.bpe import BPETokenizer
from slm.sampling.generator import TextGenerator
from slm.checkpoint.manager import CheckpointManager
from slm.utils.logger import get_logger

logger = get_logger("slm.chat")


def start_chat(checkpoint_path: str = None) -> None:
    """
    Launches an interactive console terminal chat interface.
    """
    print("\n" + "=" * 60)
    print("  SLM INTERACTIVE CHAT ENGINE (Built Completely From Scratch)")
    print("=" * 60)
    print("Type your question/prompt below. Type 'exit', 'quit', or 'q' to end session.")
    print("=" * 60 + "\n")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_dir = os.path.dirname(checkpoint_path)
        logger.info(f"Loading model checkpoint from {checkpoint_path}...")
        
        try:
            ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt_data = torch.load(checkpoint_path, map_location="cpu")
            
        if isinstance(ckpt_data, dict) and "model_config" in ckpt_data:
            config = ModelConfig.from_dict(ckpt_data["model_config"])
        else:
            config = ModelConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=2)
            
        model = SLMForCausalLM(config)
        
        manager = CheckpointManager(output_dir=ckpt_dir)
        manager.load_checkpoint(checkpoint_path, model)
        
        tok_dir = os.path.join(ckpt_dir, "tokenizer")
        if os.path.exists(tok_dir):
            tokenizer = BPETokenizer.load(tok_dir)
        else:
            tokenizer = BPETokenizer()
            tokenizer.train_on_texts(["Interactive chat training text sample for tokenizer setup."], vocab_size=config.vocab_size)
    else:
        logger.info("No checkpoint provided. Initializing active SLM model for demo chat session...")
        config = ModelConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=2, d_ff=512)
        model = SLMForCausalLM(config)
        tokenizer = BPETokenizer()
        corpus = [
            "User: What is a Small Language Model?\nSLM: A Small Language Model is an efficient decoder-only transformer network.",
            "User: How does self-attention work?\nSLM: Self-attention computes scaled dot-product matrix operations over queries, keys, and values.",
            "User: Hello!\nSLM: Hello! How can I assist you with language modeling today?"
        ] * 10
        tokenizer.train_on_texts(corpus, vocab_size=2000)

    generator = TextGenerator(model, tokenizer)

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nEnding chat session. Goodbye!")
                break

            prompt = f"User: {user_input}\nSLM:"
            print("SLM  > ", end="", flush=True)

            def stream_callback(token_str: str):
                print(token_str, end="", flush=True)

            response = generator.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                stream_callback=stream_callback
            )
            print()

        except KeyboardInterrupt:
            print("\nChat session interrupted. Goodbye!")
            break


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    start_chat(ckpt)
