#!/usr/bin/env python3
"""Push trajectories as a proper HuggingFace Dataset.

This is the script for uploading OpenHands trajectories to HuggingFace Hub.

Usage:
    1. Edit EVAL_OUTPUT_DIR and HF_REPO_ID below
    2. export HF_TOKEN="hf_..."
    3. python3 scripts/push_openhands_traj_dataset.py

The uploaded dataset can be loaded with:
    from datasets import load_dataset
    ds = load_dataset('your-repo-id')
"""

import json
from pathlib import Path
from datasets import Dataset
import os

# Configuration
EVAL_OUTPUT_DIR = Path("trajectories/SWE-bench__SWE-smith-train/CodeActAgent/gpt-4o_maxiter_50")
OUTPUT_FILE = EVAL_OUTPUT_DIR / "output.jsonl"
HF_REPO_ID = "collinear-ai/TEMP_spider_openhands_integration_test"
HF_TOKEN = os.getenv("HF_TOKEN")

def clean_dict(d):
    """Recursively clean dict, converting empty dicts to None."""
    if not isinstance(d, dict):
        return d
    
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if len(v) == 0:
                # Skip empty dicts
                continue
            else:
                cleaned[k] = clean_dict(v)
        elif isinstance(v, list):
            cleaned[k] = [clean_dict(item) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned if cleaned else None

def trajectory_to_messages(trajectory):
    """Convert OpenHands trajectory to chat format (role/content).
    
    Returns list of dicts with 'role' and 'content' keys.
    """
    messages = []
    
    for turn in trajectory:
        source = turn.get("source", "")
        action = turn.get("action", "")
        
        # Map OpenHands events to chat roles
        if source == "user":
            # User messages
            content = turn.get("args", {}).get("content", "") or turn.get("message", "")
            if content:
                messages.append({
                    "role": "user",
                    "content": content
                })
        elif source == "agent":
            # Agent actions and thoughts
            if action == "message":
                # Agent message/response
                content = turn.get("args", {}).get("content", "")
                if content:
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
            elif action in ["run", "run_ipython", "read", "write", "edit", "browse"]:
                # Agent tool use - format as assistant message
                args = turn.get("args", {})
                observation = turn.get("observation", "")
                
                # Format the action as a message
                if action == "run":
                    content = f"Running command: {args.get('command', '')}"
                    if observation:
                        content += f"\n\nOutput:\n{observation}"
                elif action == "run_ipython":
                    content = f"Running Python:\n```python\n{args.get('code', '')}\n```"
                    if observation:
                        content += f"\n\nOutput:\n{observation}"
                elif action == "read":
                    content = f"Reading file: {args.get('path', '')}"
                    if observation:
                        content += f"\n\n{observation}"
                elif action == "write":
                    content = f"Writing to file: {args.get('path', '')}\n```\n{args.get('content', '')}\n```"
                elif action == "edit":
                    content = f"Editing file: {args.get('path', '')}"
                elif action == "browse":
                    content = f"Browsing: {args.get('url', '')}"
                
                if content:
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
    
    return messages

def transform_record(record):
    """Transform to training format with messages (role/content)."""
    # Get raw trajectory
    trajectory = record.get("history", [])
    
    # Convert to messages format for training
    messages = trajectory_to_messages(trajectory)
    
    # Also keep raw trajectory for reference (cleaned)
    cleaned_trajectory = []
    for turn in trajectory:
        cleaned_turn = clean_dict(turn)
        if cleaned_turn:
            cleaned_trajectory.append(cleaned_turn)
    
    return {
        "instance_id": record.get("instance_id"),
        "repo": record.get("instance", {}).get("repo"),
        "image_name": record.get("instance", {}).get("image_name"),
        "instruction": record.get("instruction"),
        "messages": messages,  # Chat format for training!
        "trajectory": cleaned_trajectory,  # Raw trajectory for reference
        "git_patch": record.get("test_result", {}).get("git_patch") if record.get("test_result") else None,
        "agent_class": record.get("metadata", {}).get("agent_class"),
        "model": record.get("metadata", {}).get("llm_config", {}).get("model"),
        "max_iterations": record.get("metadata", {}).get("max_iterations"),
        "metrics": clean_dict(record.get("metrics", {})),
        "error": record.get("error"),
        "problem_statement": record.get("instance", {}).get("problem_statement"),
        "hints_text": record.get("instance", {}).get("hints_text"),
        "created_at": record.get("instance", {}).get("created_at"),
        "resolved": record.get("test_result", {}).get("resolved") if record.get("test_result") else None,
    }

print("="*60)
print("🚀 PUSH AS HUGGINGFACE DATASET")
print("="*60)
print()

if not OUTPUT_FILE.exists():
    print(f"❌ Error: {OUTPUT_FILE} not found")
    exit(1)

if not HF_TOKEN:
    print("❌ Error: HF_TOKEN not set")
    print("Run: export HF_TOKEN='hf_...'")
    exit(1)

# Load and transform data
print("Loading and transforming data...")
data = []
with open(OUTPUT_FILE, 'r') as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)
        transformed = transform_record(record)
        # Remove None values but keep empty lists
        transformed = {k: v for k, v in transformed.items() if v is not None}
        data.append(transformed)

print(f"  ✓ Loaded {len(data)} records")

# Show sample
print("\n" + "="*60)
print("SAMPLE RECORD:")
print("="*60)
sample = data[0]
print(f"  instance_id: {sample['instance_id']}")
print(f"  messages: list with {len(sample.get('messages', []))} messages (role/content format)")
print(f"  trajectory: list with {len(sample.get('trajectory', []))} turns (raw OpenHands events)")
print(f"  model: {sample['model']}")
print()
print("First few messages:")
for i, msg in enumerate(sample.get('messages', [])[:3]):
    print(f"  {i+1}. role={msg['role']}, content={msg['content'][:80]}...")
print()

# Create HuggingFace Dataset
print("="*60)
print("Creating HuggingFace Dataset...")
print("="*60)
print()

dataset = Dataset.from_list(data)
print(f"✓ Created dataset with {len(dataset)} rows")
print(f"\nColumn names: {dataset.column_names}")

# Push to Hub
print(f"\n{'='*60}")
print(f"Pushing to Hub: {HF_REPO_ID}")
print(f"{'='*60}\n")

try:
    dataset.push_to_hub(
        repo_id=HF_REPO_ID,
        private=True,
        token=HF_TOKEN
    )
    
    print(f"\n{'='*60}")
    print("✅ SUCCESS!")
    print(f"{'='*60}\n")
    print(f"Dataset URL: https://huggingface.co/datasets/{HF_REPO_ID}")
    print()
    print("Load it for training:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('{HF_REPO_ID}')")
    print(f"  print(ds['train'][0]['messages'])  # Chat format (role/content)")
    print()
    print("Or access raw trajectory:")
    print(f"  print(ds['train'][0]['trajectory'])  # Raw OpenHands events")
    print()
except Exception as e:
    print(f"\n❌ Error during push: {e}")
    print("\nTrying with simpler trajectory format...")
    
    # Simplify by converting trajectory to JSON string
    print("Converting trajectory to JSON strings...")
    data_simple = []
    for record in data:
        record_copy = record.copy()
        record_copy['trajectory'] = json.dumps(record['trajectory'])
        data_simple.append(record_copy)
    
    dataset_simple = Dataset.from_list(data_simple)
    print(f"✓ Created simplified dataset")
    
    dataset_simple.push_to_hub(
        repo_id=HF_REPO_ID,
        private=True,
        token=HF_TOKEN
    )
    
    print(f"\n{'='*60}")
    print("✅ SUCCESS (with trajectory as JSON string)!")
    print(f"{'='*60}\n")
    print(f"Dataset URL: https://huggingface.co/datasets/{HF_REPO_ID}")
    print()
    print("Load it:")
    print(f"  from datasets import load_dataset")
    print(f"  import json")
    print(f"  ds = load_dataset('{HF_REPO_ID}')")
    print(f"  trajectory = json.loads(ds['train'][0]['trajectory'])")
    print()
