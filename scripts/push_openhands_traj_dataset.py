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
EVAL_OUTPUT_DIR = Path("trajectories/swe-bench-train/princeton-nlp__SWE-bench-train/CodeActAgent/gpt-5_maxiter_50")
OUTPUT_FILE = EVAL_OUTPUT_DIR / "output.jsonl"
HF_REPO_ID = "collinear-ai/spider-openhands-swe-bench-trajectories"
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
    """Convert OpenHands trajectory to OpenAI function calling format.
    
    Returns list of messages in OpenAI format with tool_calls.
    Matches SWE-Gym/OpenHands-Sampled-Trajectories format.
    """
    messages = []
    i = 0
    
    # Add system message first (from first event if it's a system message)
    if trajectory and trajectory[0].get('action') == 'system':
        system_content = trajectory[0].get('args', {}).get('content', '') or trajectory[0].get('message', '')
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content,
                "function_call": None,
                "name": None,
                "tool_call_id": None,
                "tool_calls": None
            })
        i = 1  # Skip the system message
    
    while i < len(trajectory):
        turn = trajectory[i]
        source = turn.get("source", "")
        action = turn.get("action", "")
        
        # Handle user messages
        if source == "user" and action == "message":
            content = turn.get("args", {}).get("content", "") or turn.get("message", "")
            if content:
                messages.append({
                    "role": "user",
                    "content": content,
                    "function_call": None,
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": None
                })
            i += 1
            continue
        
        # Handle agent tool calls (actions)
        if source == "agent" and action in ["run", "run_ipython", "read", "write", "edit", "browse", "str_replace", "task_tracking"]:
            tool_call_metadata = turn.get('tool_call_metadata', {})
            tool_call_id = tool_call_metadata.get('tool_call_id', turn.get('id', f"call_{i}"))
            
            # Create tool call
            args = turn.get("args", {})
            tool_calls = [{
                "function": {
                    "arguments": json.dumps(args),
                    "name": action
                },
                "id": tool_call_id,
                "index": None,
                "type": "function"
            }]
            
            messages.append({
                "role": "assistant",
                "content": turn.get("message", "") or "",
                "function_call": None,
                "name": None,
                "tool_call_id": None,
                "tool_calls": tool_calls
            })
            
            # Check for observation in next event
            if i + 1 < len(trajectory):
                next_turn = trajectory[i + 1]
                if 'content' in next_turn and next_turn.get('source') in ['agent', 'environment']:
                    obs_content = next_turn.get("content", "")
                    # Limit observation length
                    if len(obs_content) > 50000:
                        obs_content = obs_content[:50000] + "\n... (output truncated)"
                    
                    messages.append({
                        "role": "tool",
                        "content": f"OBSERVATION:\n{obs_content}",
                        "function_call": None,
                        "name": action,
                        "tool_call_id": tool_call_id,
                        "tool_calls": None
                    })
                    i += 1  # Skip observation
            
            i += 1
            continue
        
        # Handle agent messages (non-tool responses)
        if source == "agent" and action == "message":
            content = turn.get("args", {}).get("content", "") or turn.get("message", "")
            if content:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "function_call": None,
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": None
                })
            i += 1
            continue
        
        # Skip other events
        i += 1
    
    return messages


def extract_tools_from_trajectory(trajectory):
    """Extract tools definition from trajectory.
    
    OpenHands includes tool definitions in the first system message.
    """
    if trajectory and trajectory[0].get('action') == 'system':
        tools = trajectory[0].get('args', {}).get('tools', [])
        if tools:
            return tools
    
    # If not in system message, return empty list
    return []

def transform_record(record):
    """Transform to OpenAI function calling format (matches SWE-Gym)."""
    # Get raw trajectory
    trajectory = record.get("history") or []

    # Skip records with missing/empty trajectories
    if not trajectory:
        return None
    
    # Convert to messages format (OpenAI function calling)
    messages = trajectory_to_messages(trajectory)
    
    # Extract tools definition
    tools = extract_tools_from_trajectory(trajectory)
    
    # Also keep raw trajectory for reference (cleaned)
    cleaned_trajectory = []
    for turn in trajectory:
        cleaned_turn = clean_dict(turn)
        if cleaned_turn:
            cleaned_trajectory.append(cleaned_turn)
    
    # Capture test_result but strip git_patch to avoid clashing with dataset git_patch
    test_result = record.get("test_result", {}) or {}
    generated_git_patch = test_result.pop("git_patch", None)

    # Get resolved from test_result.report.resolved (set by evaluation step)
    resolved = None
    if test_result and "report" in test_result:
        resolved = test_result.get("report", {}).get("resolved")
    
    # Build result in order: id -> original fields -> metadata -> resolved/test_result/git_patch -> nested copies -> messages/tools/trajectory
    result = {}

    # Core identifier
    result["instance_id"] = record.get("instance_id")

    # Flatten all original instance fields into top-level (preserve original dataset schema)
    instance_fields = record.get("instance", {}) or {}
    for k, v in instance_fields.items():
        if v is not None and k not in result:
            result[k] = v

    # Optional metadata fields
    if record.get("instance", {}).get("repo"):
        result["repo"] = record.get("instance", {}).get("repo")
    if record.get("instance", {}).get("image_name"):
        result["image_name"] = record.get("instance", {}).get("image_name")
    if record.get("instruction"):
        result["instruction"] = record.get("instruction")
    if record.get("metadata", {}).get("agent_class"):
        result["agent_class"] = record.get("metadata", {}).get("agent_class")
    if record.get("metadata", {}).get("llm_config", {}).get("model"):
        result["model"] = record.get("metadata", {}).get("llm_config", {}).get("model")
    if record.get("metadata", {}).get("max_iterations"):
        result["max_iterations"] = record.get("metadata", {}).get("max_iterations")
    cleaned_metrics = clean_dict(record.get("metrics", {}))
    if cleaned_metrics:
        result["metrics"] = cleaned_metrics
    if record.get("error"):
        result["error"] = record.get("error")
    if record.get("instance", {}).get("problem_statement"):
        result["problem_statement"] = record.get("instance", {}).get("problem_statement")

    # Resolved and test_result/git_patch
    result["resolved"] = resolved
    if generated_git_patch:
        result["generated_git_patch"] = generated_git_patch
    cleaned_test_result = clean_dict(test_result)
    if cleaned_test_result:
        result["test_result"] = cleaned_test_result

    # Preserve original instance fields (cleaned) as nested
    cleaned_instance = clean_dict(record.get("instance", {}))
    if cleaned_instance:
        result["instance"] = cleaned_instance

    # Conversational artifacts at the end
    result["messages"] = messages
    result["tools"] = tools
    result["trajectory"] = cleaned_trajectory
    
    return result

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
        if transformed is None:
            print(f"  ⚠️  Skipping record {i}: empty or missing history")
            continue
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
