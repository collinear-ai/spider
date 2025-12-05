"""Base classes for SWE agent scaffolds."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class ScaffoldConfig(BaseModel):
    """Base configuration for a scaffold."""
    
    output_dir: Union[Path, str]
    """Directory to save trajectory outputs."""
    
    max_instances: Optional[int] = None
    """Maximum number of instances to process. None = all."""
    
    num_workers: int = 1
    """Number of parallel workers."""
    
    timeout_seconds: Optional[int] = None
    """Timeout per instance in seconds. None = no timeout."""
    
    max_retries: int = 5
    """Maximum retries per instance on failure."""
    
    def __init__(self, **data):
        # Convert string paths to Path objects
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])
        super().__init__(**data)


class Scaffold(ABC):
    """Abstract base class for SWE agent scaffolds."""
    
    def __init__(self, config: ScaffoldConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def run_batch(
        self,
        dataset_name: str,
        split: str = "train",
        instance_filter: Optional[str] = None,
    ) -> Path:
        """Run the scaffold on a dataset and generate trajectories.
        
        Args:
            dataset_name: HuggingFace dataset name (e.g., "SWE-bench/SWE-smith")
            split: Dataset split to use (default: "train")
            instance_filter: Optional regex filter for instance IDs
            
        Returns:
            Path to output file/directory containing trajectories
        """
        pass
    
    @abstractmethod
    def run_single(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Run the scaffold on a single instance.
        
        Args:
            instance: Single task instance dictionary
            
        Returns:
            Trajectory output dictionary (format depends on scaffold)
        """
        pass

