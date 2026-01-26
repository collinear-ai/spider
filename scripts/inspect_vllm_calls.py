#!/usr/bin/env python3
"""Inspect vLLM call debug logs."""
import pickle
import sys
from pathlib import Path

debug_dir = Path("/home/ubuntu/spider/debug_vllm_calls")
if not debug_dir.exists():
    print("No debug_vllm_calls folder found")
    exit(1)

# Check for --detail flag
show_detail = "--detail" in sys.argv

files = sorted(debug_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
print(f"Found {len(files)} vLLM call logs\n")

for f in files[:10]:  # Show latest 10
    with open(f, "rb") as fp:
        data = pickle.load(fp)
    
    req = data.get("request", {})
    resp = data.get("response", {})
    
    print(f"=== {f.name} ===")
    print(f"  model: {req.get('model')}")
    print(f"  messages: {len(req.get('messages', []))} turns")
    print(f"  prompt_tokens: {resp.get('prompt_token_count')}")
    print(f"  completion_tokens: {resp.get('completion_token_count')}")
    print(f"  has_tool_calls: {bool(resp.get('tool_calls'))}")
    
    if show_detail:
        # Show last message (user turn)
        messages = req.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = last_msg.get("content", "")[:200]
            print(f"  last_input: {content}...")
        
        # Show response content
        content = resp.get("content", "") or ""
        print(f"  output: {content[:200]}..." if len(content) > 200 else f"  output: {content}")
        
        # Show logprobs sample
        logprobs = resp.get("logprobs", [])
        if logprobs:
            print(f"  logprobs[:5]: {logprobs[:5]}")
    print()
