"""
Command Line Interface (CLI) for training, generation, tokenization, and benchmarking.
"""

import argparse
import os
import sys
from typing import List

from slm.config.config_loader import load_config, save_config
from slm.config.model_config import ModelConfig
from slm.config.train_config import TrainConfig
from slm.dataset.cleaner import TextCleaner
from slm.dataset.readers import DatasetReader
from slm.dataset.dataset import CausalLMDataset
from slm.dataset.loader import create_dataloader
from slm.tokenizer.bpe import BPETokenizer
from slm.model.transformer_lm import SLMForCausalLM
from slm.sampling.generator import TextGenerator
from slm.training.trainer import Trainer
from slm.evaluation.benchmark import benchmark_inference, benchmark_training_step
from slm.checkpoint.manager import CheckpointManager
from slm.utils.logger import get_logger
from slm.utils.utils import set_seed, get_device

logger = get_logger("slm.cli")


def cmd_train(args: argparse.Namespace) -> None:
    """Handles 'train' CLI command."""
    logger.info(f"Loading configuration from {args.config}...")
    model_config, train_config = load_config(args.config)

    if args.dataset:
        train_config.train_dataset_path = args.dataset

    set_seed(train_config.seed)

    # Ingest text documents
    if train_config.train_dataset_path and os.path.exists(train_config.train_dataset_path):
        docs = DatasetReader.load_file(train_config.train_dataset_path)
    else:
        logger.info("No training dataset path supplied. Using synthetic demonstration corpus.")
        docs = [
            "Deep learning allows computational models composed of multiple processing layers to learn representations of data with multiple levels of abstraction.",
            "Transformer architectures rely entirely on self-attention mechanisms to compute representations of their input and output without using sequence-aligned RNNs.",
            "Byte Pair Encoding subword tokenization enables vocabulary scaling across arbitrary natural language text corpora.",
        ] * 100

    # Train or load BPE tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train_on_texts(docs, vocab_size=model_config.vocab_size)

    # Build dataset and dataloader
    dataset = CausalLMDataset(docs, tokenizer, max_seq_len=model_config.max_seq_len)
    loader = create_dataloader(dataset, batch_size=train_config.batch_size, shuffle=True)

    # Build model and trainer
    model = SLMForCausalLM(model_config)
    trainer = Trainer(model, train_config, loader, tokenizer=tokenizer)

    # Execute training loop
    summary = trainer.train()
    logger.info(f"Training finished successfully! Final checkpoint saved to: {summary['final_checkpoint']}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Handles 'generate' CLI command."""
    logger.info(f"Initializing text generator for prompt: '{args.prompt}'")
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_manager = CheckpointManager(output_dir=os.path.dirname(args.checkpoint))
        # Inspect checkpoint to get saved model_config if present
        try:
            ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except Exception:
            ckpt_data = torch.load(args.checkpoint, map_location="cpu")
        
        if isinstance(ckpt_data, dict) and "model_config" in ckpt_data:
            config = ModelConfig.from_dict(ckpt_data["model_config"])
        else:
            config = ModelConfig(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
            
        model = SLMForCausalLM(config)
        ckpt_manager.load_checkpoint(args.checkpoint, model)
        tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(args.checkpoint), "tokenizer"))
    else:
        logger.info("No checkpoint provided. Initializing lightweight untrained SLM model for demo.")
        config = ModelConfig(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=512)
        model = SLMForCausalLM(config)
        tokenizer = BPETokenizer()
        tokenizer.train_on_texts([args.prompt, "Language model generation testing completely from scratch."], vocab_size=1000)

    generator = TextGenerator(model, tokenizer)
    output = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print("\n" + "=" * 50)
    print("  GENERATED TEXT OUTPUT")
    print("=" * 50)
    print(output)
    print("=" * 50 + "\n")


def cmd_tokenize(args: argparse.Namespace) -> None:
    """Handles 'tokenize' CLI command."""
    tokenizer = BPETokenizer()
    tokenizer.train_on_texts([args.text], vocab_size=500)
    token_ids = tokenizer.encode(args.text)
    decoded = tokenizer.decode(token_ids)

    print("\n" + "=" * 50)
    print(f"  Input Text:   {args.text}")
    print(f"  Token IDs:    {token_ids}")
    print(f"  Decoded Text: {decoded}")
    print("=" * 50 + "\n")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handles 'benchmark' CLI command."""
    device = get_device(args.device)
    logger.info(f"Running SLM Benchmark on device={device}...")

    config = ModelConfig(vocab_size=10000, d_model=256, n_heads=8, n_layers=4, d_ff=1024)
    model = SLMForCausalLM(config)

    res = benchmark_inference(model, batch_size=args.batch_size, seq_len=args.seq_len, device=device)
    
    print("\n" + "=" * 50)
    print("  SLM INFERENCE BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Device:          {res['device']}")
    print(f"  Batch Size:      {res['batch_size']}")
    print(f"  Sequence Length: {res['seq_len']}")
    print(f"  Latency (ms):    {res['avg_latency_ms']} ms")
    print(f"  Throughput:      {res['tokens_per_sec']} tokens/sec")
    print("=" * 50 + "\n")


def cmd_chat(args: argparse.Namespace) -> None:
    """Handles 'chat' CLI command."""
    from scripts.chat_run import start_chat
    start_chat(args.checkpoint)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="SLM CLI - Small Language Model Tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", type=str, default="configs/nano_config.yaml", help="Path to config file")
    train_parser.add_argument("--dataset", type=str, default=None, help="Path to training dataset file")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Input prompt string")
    gen_parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    gen_parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top_k", type=int, default=40, help="Top-K limit")
    gen_parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus threshold")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat REPL")
    chat_parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")

    # Tokenize command
    tok_parser = subparsers.add_parser("tokenize", help="Tokenize text")
    tok_parser.add_argument("--text", type=str, required=True, help="Text to tokenize")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    bench_parser.add_argument("--seq_len", type=int, default=256, help="Sequence context length")
    bench_parser.add_argument("--device", type=str, default="auto", help="Device choice")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "tokenize":
        cmd_tokenize(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
