from __future__ import annotations
import typer
from .client import SyntheticDataClient

app = typer.Typer(help="Synthetic data generation client")

@app.command()
def run(config: str):
    raise NotImplementedError