#!/usr/bin/env python3
"""Inspect loss debug logs."""
import pickle
from pathlib import Path

debug_dir = Path("/home/ubuntu/spider/debug_loss")
if not debug_dir.exists():
    print("No debug_loss folder found")
    exit(1)

files = sorted(debug_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
print(f"Found {len(files)} loss debug logs\n")

for f in files[:5]:  # Show latest 5
    with open(f, "rb") as fp:
        data = pickle.load(fp)
    
    print(f"=== {f.name} ===")
    print(f"  batch_index: {data.get('batch_index')}")
    print(f"  trajectory_idx: {data.get('trajectory_idx')}")
    print(f"  loss: {data.get('loss')}")
    print(f"  clip_ratio: {data.get('clip_ratio')}")
    print(f"  metrics: {data.get('metrics')}")
    
    input_ids = data.get('input_ids')
    if input_ids is not None:
        print(f"  input_ids shape: {input_ids.shape}")
    
    sampling_lp = data.get('sampling_logprobs')
    current_lp = data.get('current_logprobs')
    if sampling_lp is not None and current_lp is not None:
        print(f"  sampling_logprobs: min={sampling_lp.min():.4f} max={sampling_lp.max():.4f} mean={sampling_lp.mean():.4f}")
        print(f"  current_logprobs: min={current_lp.min():.4f} max={current_lp.max():.4f} mean={current_lp.mean():.4f}")
        ratio = (current_lp - sampling_lp).exp()
        print(f"  importance ratio: min={ratio.min():.4f} max={ratio.max():.4f} mean={ratio.mean():.4f}")
    
    advantages = data.get('advantages')
    if advantages is not None:
        print(f"  advantages: min={advantages.min():.4f} max={advantages.max():.4f} mean={advantages.mean():.4f}")
    
    loss_mask = data.get('loss_mask')
    if loss_mask is not None:
        print(f"  loss_mask: sum={loss_mask.sum().item():.0f} / {loss_mask.numel()}")
    print()
