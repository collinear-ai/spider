"""CLI for running SWE scaffolds."""

import json
import sys
from pathlib import Path

import typer
import yaml
from rich import print as rprint

try:
    from server.scaffolds.openhands_wrapper import (
        OpenHandsScaffold,
        OpenHandsScaffoldConfig,
    )
except ImportError as e:
    rprint(f"[red]Error: Could not import OpenHands scaffold: {e}[/red]")
    rprint("[yellow]Make sure you have installed: pip install spider[swe-scaffolds][/yellow]")
    sys.exit(1)

app = typer.Typer(help="SWE scaffold trajectory generation")


@app.command("openhands")
def openhands(
    config_path: Path = typer.Option(..., "--config", "-c", help="Config file path"),
    dataset: str = typer.Option(None, "--dataset", help="Override dataset name"),
    split: str = typer.Option(None, "--split", help="Override dataset split"),
    max_instances: int = typer.Option(None, "--max-instances", help="Override max instances"),
):
    """Run OpenHands scaffold to generate trajectories."""
    # Load config
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    scaffold_config = config_data.get("scaffold", {})
    
    # Override with CLI args if provided
    if dataset:
        scaffold_config["dataset"] = dataset
    if split:
        scaffold_config["split"] = split
    if max_instances:
        scaffold_config["max_instances"] = max_instances
    
    # Create config object
    config = OpenHandsScaffoldConfig(**scaffold_config)
    
    rprint(f"[green]Starting OpenHands trajectory generation[/green]")
    rprint(f"  Dataset: {scaffold_config.get('dataset', 'N/A')}")
    rprint(f"  Split: {scaffold_config.get('split', 'train')}")
    rprint(f"  Output: {config.output_dir}")
    
    # Create scaffold and run
    scaffold = OpenHandsScaffold(config)
    
    dataset_name = scaffold_config.get("dataset", "SWE-bench/SWE-smith")
    split_name = scaffold_config.get("split", "train")
    instance_filter = scaffold_config.get("instance_filter")
    
    output_path = scaffold.run_batch(
        dataset_name=dataset_name,
        split=split_name,
        instance_filter=instance_filter,
    )
    
    rprint(f"[green]✓ Trajectories saved to: {output_path}[/green]")


if __name__ == "__main__":
    app()

