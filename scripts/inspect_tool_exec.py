#!/usr/bin/env python3
"""Inspect tool execution debug logs."""
import json
from pathlib import Path

debug_dir = Path("/home/ubuntu/spider/debug_tool_exec")
if not debug_dir.exists():
    print("No debug_tool_exec folder found")
    exit(1)

files = sorted(debug_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
print(f"Found {len(files)} tool execution logs\n")

for f in files[:10]:  # Show latest 10
    data = json.loads(f.read_text())
    print(f"=== {f.name} ===")
    print(f"  prompt_idx: {data.get('prompt_idx', 'N/A')}")
    print(f"  turn_idx: {data.get('turn_idx', 'N/A')}")
    print(f"  tool: {data.get('tool_name')}")
    print(f"  duration: {data.get('duration_s')}s")
    print(f"  error: {data.get('error')}")
    print(f"  args: {str(data.get('args'))[:100]}...")
    result = data.get('result', '')
    print(f"  result: {result[:200]}..." if len(result) > 200 else f"  result: {result}")
    print()
