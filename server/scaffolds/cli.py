"""CLI for SWE scaffolds."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

import typer

try:
    from server.scaffolds.openhands_wrapper import (
        OpenHandsScaffold,
        OpenHandsScaffoldConfig,
    )
    OPENHANDS_AVAILABLE = True
except ImportError:
    OPENHANDS_AVAILABLE = False

# TODO: Add swe-agent when implemented
# try:
#     from server.scaffolds.swe_agent_wrapper import (
#         SWEAgentScaffold,
#         SWEAgentScaffoldConfig,
#     )
#     SWE_AGENT_AVAILABLE = True
# except ImportError:
#     SWE_AGENT_AVAILABLE = False

app = typer.Typer(help="SWE trajectory generation scaffolds")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """SWE trajectory generation scaffolds."""
    if ctx.invoked_subcommand is None:
        typer.echo("Error: Please specify a command")
        typer.echo("Run 'spider-scaffold --help' for usage information.")
        raise typer.Exit(1)


def _create_scaffold(scaffold_type: str, config_dict: Dict[str, Any]):
    """Factory function to create scaffold instances based on type.
    
    Args:
        scaffold_type: Type of scaffold ('openhands', 'swe-agent', etc.)
        config_dict: Configuration dictionary for the scaffold
        
    Returns:
        Scaffold instance
        
    Raises:
        ValueError: If scaffold type is not supported
        ImportError: If scaffold dependencies are not available
    """
    if scaffold_type == "openhands":
        if not OPENHANDS_AVAILABLE:
            raise ImportError(
                "OpenHands scaffold not available. "
                "Install with: pip install spider[swe-scaffolds]"
            )
        
        # Handle API key from environment variable if specified
        if "llm_api_key_env" in config_dict and config_dict["llm_api_key_env"]:
            env_var = config_dict["llm_api_key_env"]
            api_key = os.getenv(env_var)
            if api_key:
                config_dict["llm_api_key"] = api_key
            elif not config_dict.get("llm_api_key"):
                typer.echo(
                    f"Warning: Environment variable {env_var} not set, and no llm_api_key in config",
                    err=True,
                )
        
        oh_config = OpenHandsScaffoldConfig(**config_dict)
        return OpenHandsScaffold(oh_config)
    
    elif scaffold_type == "swe-agent":
        # TODO: When swe-agent is implemented, uncomment and add:
        # if not SWE_AGENT_AVAILABLE:
        #     raise ImportError("SWE-agent scaffold not available. Install dependencies...")
        # swe_config = SWEAgentScaffoldConfig(**config_dict)
        # return SWEAgentScaffold(swe_config)
        raise NotImplementedError(
            "SWE-agent scaffold is not yet implemented. "
            "Currently supported: openhands"
        )
    
    elif scaffold_type == "mini-swe-agent":
        raise NotImplementedError(
            "mini-swe-agent scaffold is not yet implemented. "
            "Currently supported: openhands"
        )
    
    else:
        raise ValueError(
            f"Unknown scaffold type: {scaffold_type}. "
            f"Supported types: openhands (swe-agent and mini-swe-agent coming soon)"
        )


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    """Run a scaffold to generate SWE trajectories.
    
    The scaffold type is determined from the config file's 'scaffold.type' field.
    Supported scaffolds: openhands (swe-agent and mini-swe-agent coming soon)
    """
    # Load config file
    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    # Extract scaffold config
    if "scaffold" not in config_data:
        typer.echo("Error: Config file must have 'scaffold' key", err=True)
        raise typer.Exit(1)
    
    scaffold_config_dict = config_data["scaffold"].copy()
    
    # Merge root-level config fields into scaffold config (for convenience)
    # This allows users to put fields at root level instead of nested under scaffold
    for key, value in config_data.items():
        if key != "scaffold" and key not in scaffold_config_dict:
            scaffold_config_dict[key] = value
    
    scaffold_type = scaffold_config_dict.get("type")
    
    if not scaffold_type:
        typer.echo("Error: Config file must specify 'scaffold.type'", err=True)
        raise typer.Exit(1)
    
    # Create scaffold and run
    try:
        scaffold = _create_scaffold(scaffold_type, scaffold_config_dict)
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except NotImplementedError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error creating scaffold: {e}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Starting {scaffold_type} scaffold with config: {config_path}")
    typer.echo(f"Dataset: {scaffold_config_dict.get('dataset', 'N/A')}")
    typer.echo(f"Output directory: {scaffold_config_dict.get('output_dir', 'N/A')}")
    
    try:
        output_path = scaffold.run_batch()
        typer.echo(f"\n✓ Trajectories saved to: {output_path}")
    except Exception as e:
        typer.echo(f"Error running scaffold: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def openhands(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    """Run OpenHands scaffold to generate trajectories.
    
    This is a convenience command. You can also use 'run' and specify the scaffold type in the config.
    """
    if not OPENHANDS_AVAILABLE:
        typer.echo(
            "Error: OpenHands scaffold not available. "
            "Install with: pip install spider[swe-scaffolds]",
            err=True,
        )
        raise typer.Exit(1)
    
    # Load config file
    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    # Extract scaffold config
    if "scaffold" not in config_data:
        typer.echo("Error: Config file must have 'scaffold' key", err=True)
        raise typer.Exit(1)
    
    scaffold_config_dict = config_data["scaffold"].copy()
    
    # Merge root-level config fields into scaffold config (for convenience)
    # This allows users to put fields at root level instead of nested under scaffold
    for key, value in config_data.items():
        if key != "scaffold" and key not in scaffold_config_dict:
            scaffold_config_dict[key] = value
    
    # Ensure type is openhands (or set it if not specified)
    if scaffold_config_dict.get("type") and scaffold_config_dict.get("type") != "openhands":
        typer.echo(
            f"Warning: Config specifies scaffold type '{scaffold_config_dict.get('type')}', "
            f"but using 'openhands' command. Overriding to 'openhands'.",
            err=True,
        )
    scaffold_config_dict["type"] = "openhands"
    
    # Create scaffold and run using the factory
    try:
        scaffold = _create_scaffold("openhands", scaffold_config_dict)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Starting OpenHands scaffold with config: {config_path}")
    typer.echo(f"Dataset: {scaffold_config_dict.get('dataset', 'N/A')}")
    typer.echo(f"Output directory: {scaffold_config_dict.get('output_dir', 'N/A')}")
    
    try:
        output_path = scaffold.run_batch()
        typer.echo(f"\n✓ Trajectories saved to: {output_path}")
    except Exception as e:
        typer.echo(f"Error running scaffold: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

