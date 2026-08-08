"""
Hugging Face Hub Uploader Script for LawSLM.

Uploads model weights, tokenizer, configs, and repository card (README.md & MODEL_CARD.md)
to Hugging Face Hub while filtering out build/node artifacts.
"""

import argparse
import os
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Package 'huggingface_hub' is required. Install using: pip install huggingface_hub")
    sys.exit(1)


def push_to_huggingface(repo_id: str, local_dir: str = ".", token: str = None, private: bool = False) -> str:
    """
    Pushes project repository and model card to Hugging Face Hub.
    """
    # Clean token string if wrapped in quotes
    if token:
        token = token.strip('"').strip("'").strip()
    else:
        token = os.getenv("HF_TOKEN")
        if token:
            token = token.strip('"').strip("'").strip()

    if "your-username" in repo_id:
        print("\n[ERROR] 'your-username' is a placeholder!")
        print("Please replace 'your-username' with your real Hugging Face account username.")
        print("Example: python scripts/push_to_hf.py --repo_id Amit123103/Law_model_slm --token hf_XXXXX\n")
        sys.exit(1)

    api = HfApi(token=token)

    # Auto-resolve username if only repo name is supplied
    if "/" not in repo_id and token:
        try:
            user_info = api.whoami(token=token)
            username = user_info.get("name")
            if username:
                repo_id = f"{username}/{repo_id}"
                print(f"Auto-resolved Hugging Face repository ID to: '{repo_id}'")
        except Exception:
            pass

    print(f"Creating/verifying Hugging Face repository '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    
    # Patterns to ignore when uploading to HF Hub (prevents unsafe file warnings)
    ignore_patterns = [
        "node_modules/*",
        "**/node_modules/*",
        "dist/*",
        "**/dist/*",
        ".oxlintrc.json",
        ".pytest_cache/*",
        "__pycache__/*",
        "**/__pycache__/*",
        "*.pyc",
        ".env",
        ".git/*",
        "*.log"
    ]
    
    print(f"Uploading repository '{local_dir}' to Hugging Face Hub ({repo_id})...")
    url = api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
        token=token
    )
    
    print(f"\nSuccessfully pushed model repository to Hugging Face!")
    print(f"Repository URL: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(description="Push LawSLM code, config, and Model Card to Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., 'Amit123103/Law_model_slm')")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face User Access Token (Write permission)")
    parser.add_argument("--private", action="store_true", help="Create as private repository")
    args = parser.parse_args()

    push_to_huggingface(repo_id=args.repo_id, token=args.token, private=args.private)


if __name__ == "__main__":
    main()
