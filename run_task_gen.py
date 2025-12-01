#!/usr/bin/env python3
"""Quick script to run SWE task generation"""

import sys
from pathlib import Path
from spider.client import SpiderClient
from spider.config import AppConfig

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_task_gen.py <config.yaml>")
        print("\nExample:")
        print("  python run_task_gen.py config/swe/minimal-example.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading config: {config_path}")
    config = AppConfig.load(config_path)
    
    print("Submitting job...")
    with SpiderClient(config=config) as client:
        submission = client.submit_job()
        job_id = submission["job_id"]
        print(f"✓ Job submitted: {job_id}")
        print(f"  Monitor at: {config.server.base_url}/v1/jobs/{job_id}")
        print("\nWaiting for completion...")
        
        last_event_key = None
        
        def on_update(status):
            nonlocal last_event_key
            status_str = status.get("status", "unknown")
            # Print status on a single line
            print(f"  Status: {status_str}", end="\r")
            
            events = status.get("events") or []
            if not events:
                return

            # Only show the latest event, and only if it's new
            latest = events[-1]
            event_key = latest.get("code", "") + latest.get("message", "")
            if event_key == last_event_key:
                return
            last_event_key = event_key

            msg = latest.get("message", "")
            level = latest.get("level", "info").upper()
            if not msg:
                return

            # Move to new line before printing the event, keep format compact
            if level == "ERROR":
                print(f"\n  ✗ {msg}")
            elif level == "WARNING":
                print(f"\n  ⚠ {msg}")
            else:
                print(f"\n  → {msg}")
        
        try:
            status = client.poll_job(
                job_id,
                interval=10.0,
                timeout=3600,
                wait_for_completion=True,
                on_update=on_update
            )
            
            print(f"\n\nFinal status: {status['status']}")
            
            if status["status"] == "completed":
                output_path = f"./tasks_{job_id}.jsonl"
                client.download_result(job_id, destination=output_path)
                print(f"✓ Tasks saved to: {output_path}")
                
                # Show summary
                import json
                with open(output_path) as f:
                    if output_path.endswith('.jsonl'):
                        count = sum(1 for _ in f)
                    else:
                        data = json.load(f)
                        count = len(data) if isinstance(data, list) else 1
                print(f"✓ Generated {count} task instances")
            else:
                error = status.get("error_message", "Unknown error")
                print(f"✗ Job failed: {error}")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Job is still running on server.")
            print(f"Check status: {config.server.base_url}/v1/jobs/{job_id}")
            sys.exit(1)

if __name__ == "__main__":
    main()
