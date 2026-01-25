#!/usr/bin/env python3
"""Push a LoRA checkpoint step to Hugging Face Hub.

Usage:
    python scripts/push_lora_to_hub.py --step 38 --repo-id your-username/model-name
    
    # Push merged model (LoRA weights merged into base model)
    python scripts/push_lora_to_hub.py --step 38 --repo-id your-username/model-name --merge
    
    # Push just the LoRA adapter
    python scripts/push_lora_to_hub.py --step 38 --repo-id your-username/model-name --adapter-only
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Push LoRA checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step number to push (e.g., 38 for step_38)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., 'your-username/model-name')",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="/home/ubuntu/spider/workspace_accelerate_other_one",
        help="Path to workspace directory containing training/lora_checkpoints",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA weights into base model before pushing (creates full model)",
    )
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Push only the LoRA adapter files (default behavior if --merge not specified)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    workspace = Path(args.workspace)
    checkpoint_dir = workspace / "training" / "lora_checkpoints" / f"step_{args.step}"
    
    if not checkpoint_dir.exists():
        logger.error("Checkpoint directory not found: %s", checkpoint_dir)
        available_steps = sorted([
            int(d.name.replace("step_", ""))
            for d in (workspace / "training" / "lora_checkpoints").iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ])
        logger.info("Available steps: %s", available_steps)
        sys.exit(1)
    
    # Load adapter config to get base model
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config["base_model_name_or_path"]
    logger.info("Base model: %s", base_model_name)
    logger.info("LoRA checkpoint: %s", checkpoint_dir)
    
    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Create or get repo
    api = HfApi()
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        logger.info("Repository ready: %s", args.repo_id)
    except Exception as e:
        logger.warning("Could not create repo (may already exist): %s", e)
    
    commit_message = args.commit_message or f"Upload LoRA checkpoint step_{args.step}"
    
    if args.merge:
        # Merge LoRA into base model and push full model
        logger.info("Loading base model for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        
        logger.info("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        
        logger.info("Pushing merged model to Hub...")
        merged_model.push_to_hub(
            args.repo_id,
            commit_message=commit_message,
            private=args.private,
        )
        tokenizer.push_to_hub(
            args.repo_id,
            commit_message=commit_message,
            private=args.private,
        )
        
        logger.info("✅ Merged model pushed successfully to: https://huggingface.co/%s", args.repo_id)
        
    else:
        # Push adapter only
        logger.info("Pushing LoRA adapter files to Hub...")
        
        # Upload adapter files
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=args.repo_id,
            commit_message=commit_message,
        )
        
        # Also push tokenizer for convenience
        logger.info("Loading and pushing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        tokenizer.push_to_hub(
            args.repo_id,
            commit_message=f"Add tokenizer from {base_model_name}",
            private=args.private,
        )
        
        logger.info("✅ LoRA adapter pushed successfully to: https://huggingface.co/%s", args.repo_id)
        logger.info("")
        logger.info("To load this adapter:")
        logger.info("  from peft import PeftModel")
        logger.info("  from transformers import AutoModelForCausalLM")
        logger.info("")
        logger.info("  base_model = AutoModelForCausalLM.from_pretrained('%s')", base_model_name)
        logger.info("  model = PeftModel.from_pretrained(base_model, '%s')", args.repo_id)


if __name__ == "__main__":
    main()
