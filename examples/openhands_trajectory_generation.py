#!/usr/bin/env python3
"""Example: Generate SFT trajectories using OpenHands scaffold.

This example shows how to use the OpenHands scaffold to generate
trajectories from SWE task datasets for training.
"""

from pathlib import Path

from spider.server.scaffolds.openhands_wrapper import (
    OpenHandsScaffold,
    OpenHandsScaffoldConfig,
)


def main():
    """Generate trajectories from SWE-smith dataset."""
    # Create configuration
    config = OpenHandsScaffoldConfig(
        output_dir=Path("./trajectories/openhands"),
        dataset="SWE-bench/SWE-smith",
        split="train",
        max_instances=5,  # Limit to 5 for testing
        agent_class="CodeActAgent",
        max_iterations=50,
        llm_model="anthropic/claude-sonnet-4",  # Or use llm_config_name
        num_workers=1,
        timeout_seconds=3600,  # 1 hour per instance
        max_retries=3,
    )
    
    # Create scaffold
    scaffold = OpenHandsScaffold(config)
    
    # Run batch generation
    print(f"Generating trajectories from {config.dataset}...")
    output_path = scaffold.run_batch()
    
    print(f"\n✓ Trajectories saved to: {output_path}")
    print(f"\nYou can now use these trajectories for SFT training.")
    print(f"Each line in the JSONL file contains:")
    print(f"  - instance_id: Task identifier")
    print(f"  - instruction: Task description")
    print(f"  - history: Full trajectory of agent actions")
    print(f"  - test_result: Git patch with changes")
    print(f"  - metrics: Evaluation metrics")


if __name__ == "__main__":
    main()

