#!/usr/bin/env python3
"""Inspect rollout debug logs."""
import pickle
import sys
from pathlib import Path

debug_dir = Path("/home/ubuntu/spider/debug_chunk_history")
if not debug_dir.exists():
    print("No debug_chunk_history folder found")
    exit(1)

# Check for --detail flag to show full data
show_detail = "--detail" in sys.argv

files = sorted(debug_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
print(f"Found {len(files)} rollout debug logs\n")

for f in files[:10]:  # Show latest 10
    with open(f, "rb") as fp:
        data = pickle.load(fp)
    
    print(f"=== {f.name} ===")
    print(f"  prompt_prefix: {data.get('prompt_prefix')}")
    print(f"  num_turns: {data.get('num_turns')}")
    
    turns = data.get('turns', [])
    for i, t in enumerate(turns[:5]):  # Show first 5 turns
        print(f"    turn {t.get('turn_idx')}: tokens={t.get('token_count')} reward={t.get('reward_tokens')} prompt={t.get('prompt_tokens')} msgs={t.get('messages_count')}")
        if show_detail:
            logprobs = t.get('logprobs', [])
            print(f"      logprobs[:5]: {logprobs[:5] if logprobs else 'N/A'}")
            print(f"      parser_fallback: {t.get('parser_fallback')}")
            content = t.get('assistant_content', '') or ''
            print(f"      content: {content[:100]}..." if len(content) > 100 else f"      content: {content}")
            tool_calls = t.get('assistant_tool_calls')
            if tool_calls:
                print(f"      tool_calls: {len(tool_calls)} calls")
    if len(turns) > 5:
        print(f"    ... and {len(turns) - 5} more turns")
    print()
