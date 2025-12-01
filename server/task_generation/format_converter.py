"""Format conversion utilities for SWE tasks.

Converts between SWE-smith format and HF dataset format.
"""

from __future__ import annotations

from typing import Any, Dict, List


def convert_swesmith_to_hf_format(swesmith_task: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SWE-smith task format to HF dataset format.
    
    SWE-smith format:
    {
        "instance_id": "...",
        "repo": "swesmith/owner__repo.commit",
        "image_name": "jyangballin/swesmith.x86_64.owner__repo.commit",
        "patch": "git diff...",
        "FAIL_TO_PASS": ["test::test_name"],
        "PASS_TO_PASS": ["test::test_other"],
        "problem_statement": "...",
        "base_commit": "..."
    }
    
    HF dataset format (Multi-SWE-bench style):
    {
        "org": "...",
        "repo": "...",
        "instance_id": "...",
        "base": {...},
        "fix_patch": "...",
        "test_patch": "...",
        "f2p_tests": {"test": [...]},
        "p2p_tests": {"test": [...]},
        ...
    }
    """
    # Extract org and repo from SWE-smith format
    repo_str = swesmith_task.get("repo", "")
    if "/" in repo_str:
        org, repo_with_commit = repo_str.split("/", 1)
        # Remove .commit suffix if present
        repo_name = repo_with_commit.split(".")[0]
    else:
        # Fallback: try to extract from instance_id
        instance_id = swesmith_task.get("instance_id", "")
        parts = instance_id.split("__")
        if len(parts) >= 2:
            org = parts[0]
            repo_name = parts[1].split(".")[0]
        else:
            org = "unknown"
            repo_name = repo_str.replace("__", "_")
    
    # Convert test formats
    f2p_tests = swesmith_task.get("FAIL_TO_PASS", [])
    p2p_tests = swesmith_task.get("PASS_TO_PASS", [])
    
    hf_task = {
        "org": org,
        "repo": repo_name,
        "instance_id": swesmith_task.get("instance_id", ""),
        "base": {
            "commit": swesmith_task.get("base_commit", "HEAD"),
            "repo": swesmith_task.get("repo", ""),
        },
        "fix_patch": swesmith_task.get("patch", ""),
        "test_patch": swesmith_task.get("test_patch", ""),
        "problem_statement": swesmith_task.get("problem_statement", ""),
        "f2p_tests": {
            "test": f2p_tests if isinstance(f2p_tests, list) else []
        },
        "p2p_tests": {
            "test": p2p_tests if isinstance(p2p_tests, list) else []
        },
        "s2p_tests": swesmith_task.get("s2p_tests", {}),
        "n2p_tests": swesmith_task.get("n2p_tests", {}),
        "fixed_tests": swesmith_task.get("fixed_tests", {}),
        "resolved_issues": swesmith_task.get("resolved_issues", []),
        "run_result": swesmith_task.get("run_result", {}),
        "test_patch_result": swesmith_task.get("test_patch_result", {}),
        "fix_patch_result": swesmith_task.get("fix_patch_result", {}),
        # Keep SWE-smith specific fields for compatibility
        "image_name": swesmith_task.get("image_name"),
        "swesmith_metadata": {
            "bug_type": swesmith_task.get("bug_type"),
            "generation_method": swesmith_task.get("generation_method"),
            "repo": swesmith_task.get("repo"),
            "base_commit": swesmith_task.get("base_commit"),
        }
    }
    
    return hf_task


def convert_hf_to_swesmith_format(hf_task: Dict[str, Any]) -> Dict[str, Any]:
    """Convert HF dataset format back to SWE-smith format.
    
    Useful for loading tasks from HF datasets and using with scaffolds.
    """
    swesmith_task = {
        "instance_id": hf_task.get("instance_id", ""),
        "repo": hf_task.get("base", {}).get("repo", f"{hf_task.get('org', 'unknown')}/{hf_task.get('repo', 'unknown')}"),
        "image_name": hf_task.get("image_name"),
        "patch": hf_task.get("fix_patch", ""),
        "test_patch": hf_task.get("test_patch", ""),
        "FAIL_TO_PASS": hf_task.get("f2p_tests", {}).get("test", []),
        "PASS_TO_PASS": hf_task.get("p2p_tests", {}).get("test", []),
        "problem_statement": hf_task.get("problem_statement", ""),
        "base_commit": hf_task.get("base", {}).get("commit", "HEAD"),
        "s2p_tests": hf_task.get("s2p_tests", {}),
        "n2p_tests": hf_task.get("n2p_tests", {}),
        "fixed_tests": hf_task.get("fixed_tests", {}),
        "resolved_issues": hf_task.get("resolved_issues", []),
        "run_result": hf_task.get("run_result", {}),
        "test_patch_result": hf_task.get("test_patch_result", {}),
        "fix_patch_result": hf_task.get("fix_patch_result", {}),
    }
    
    # Preserve any additional metadata
    if "swesmith_metadata" in hf_task:
        swesmith_task.update(hf_task["swesmith_metadata"])
    
    return swesmith_task

