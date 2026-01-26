#!/usr/bin/env python3
"""Push a LoRA checkpoint to HuggingFace Hub.

Usage:
    python scripts/push_lora_to_hub.py <checkpoint_path> <repo_id> [--merge] [--private]
    
Examples:
    # Push just the adapter
    python scripts/push_lora_to_hub.py /mnt/local/workspace_accelerate/training/lora_checkpoints/step_59 collinear-ai/qwen3-14b-stage2-step59
    
    # Merge with base model and push full model
    python scripts/push_lora_to_hub.py /mnt/local/workspace_accelerate/training/lora_checkpoints/step_59 collinear-ai/qwen3-14b-stage2-step59 --merge
    
    # Push as private repo
    python scripts/push_lora_to_hub.py /mnt/local/workspace_accelerate/training/lora_checkpoints/step_59 collinear-ai/qwen3-14b-stage2-step59 --private
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Push LoRA checkpoint to HuggingFace Hub")
    parser.add_argument("checkpoint_path", type=str, help="Path to the LoRA checkpoint directory")
    parser.add_argument("repo_id", type=str, help="HuggingFace repo ID (e.g., 'collinear-ai/model-name')")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA with base model before pushing")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--branch", type=str, default="main", help="Branch to push to")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    # Check for required files
    adapter_config = checkpoint_path / "adapter_config.json"
    adapter_model = checkpoint_path / "adapter_model.safetensors"
    
    if not adapter_config.exists():
        print(f"Error: adapter_config.json not found in {checkpoint_path}")
        sys.exit(1)
    
    if not adapter_model.exists():
        print(f"Error: adapter_model.safetensors not found in {checkpoint_path}")
        sys.exit(1)

    # Load adapter config to get base model
    with open(adapter_config) as f:
        config = json.load(f)
    
    base_model = config.get("base_model_name_or_path")
    print(f"Base model: {base_model}")
    print(f"LoRA rank: {config.get('r')}")
    print(f"LoRA alpha: {config.get('lora_alpha')}")
    print(f"Target modules: {config.get('target_modules')}")
    print()

    from huggingface_hub import HfApi, create_repo
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    if args.merge:
        print("\n=== Merging LoRA with base model ===")
        print(f"Loading base model: {base_model}")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        
        print(f"Pushing merged model to: {args.repo_id}")
        model.push_to_hub(args.repo_id, private=args.private)
        tokenizer.push_to_hub(args.repo_id, private=args.private)
        
        print(f"\n✓ Merged model pushed to: https://huggingface.co/{args.repo_id}")
    else:
        print("\n=== Pushing LoRA adapter only ===")
        
        # Upload adapter files directly
        files_to_upload = [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]
        
        # Also upload README if it exists
        readme = checkpoint_path / "README.md"
        if readme.exists():
            files_to_upload.append("README.md")
        
        for filename in files_to_upload:
            filepath = checkpoint_path / filename
            if filepath.exists():
                print(f"Uploading: {filename}")
                api.upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=filename,
                    repo_id=args.repo_id,
                    revision=args.branch,
                )
        
        print(f"\n✓ LoRA adapter pushed to: https://huggingface.co/{args.repo_id}")
        print(f"\nTo use this adapter:")
        print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype=torch.bfloat16)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "{args.repo_id}")

# Optional: Merge for faster inference
model = model.merge_and_unload()
""")


if __name__ == "__main__":
    main()
