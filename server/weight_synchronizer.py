"""Weight synchronization between training and vLLM inference.

This module provides WeightSynchronizer for syncing LoRA adapter weights
between the training model and a vLLM inference server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import httpx
import torch

if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import PeftModel
    from .backends.vllm_backend import VLLMBackend

logger = logging.getLogger(__name__)


class WeightSynchronizer:
    """Synchronize LoRA weights between training and vLLM inference.

    This class handles saving LoRA adapter checkpoints and reloading them
    into a running vLLM server using vLLM's dynamic LoRA loading API.

    Requires:
    - vLLM server started with --enable-lora
    - VLLM_ALLOW_RUNTIME_LORA_UPDATING=True environment variable
    """

    def __init__(
        self,
        sync_every_n_steps: int,
        checkpoint_dir: Path,
        vllm_base_url: str,
        lora_name: str = "student",
        vllm_backend: Optional["VLLMBackend"] = None,
        save_every_n_steps: int = 5,  # How often to persist checkpoints
    ):
        """Initialize the weight synchronizer.

        Args:
            sync_every_n_steps: How often to sync weights to vLLM (every N training steps)
            checkpoint_dir: Directory to save adapter checkpoints
            vllm_base_url: Base URL of the vLLM server
            lora_name: Name to use for the LoRA adapter in vLLM
            vllm_backend: Optional VLLMBackend instance to use directly
            save_every_n_steps: How often to save permanent checkpoints (every N steps)
        """
        self._sync_steps = sync_every_n_steps
        self._save_steps = save_every_n_steps
        self._checkpoint_dir = Path(checkpoint_dir)
        self._lora_name = lora_name
        self._vllm_backend = vllm_backend
        self._last_sync_step = -1
        self._last_save_step = -1
        self._adapter_loaded = False

        if vllm_backend is not None:
            self._client = None  # Use backend directly
        else:
            self._client = httpx.Client(
                base_url=vllm_base_url,
                timeout=httpx.Timeout(60.0, connect=30.0),
            )

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._current_checkpoint = self._checkpoint_dir / "current"
        self._current_checkpoint.mkdir(parents=True, exist_ok=True)

    def maybe_sync(
        self,
        model: "PeftModel",
        accelerator: "Accelerator",
        step: int,
    ) -> bool:
        """Save checkpoint and reload vLLM if at sync interval.

        Args:
            model: PEFT model with LoRA adapter
            accelerator: Accelerator instance for unwrapping model
            step: Current training step

        Returns:
            True if sync was performed, False otherwise
        """
        if step <= self._last_sync_step:
            return False

        if step % self._sync_steps != 0:
            return False

        return self.force_sync(model, accelerator, step)

    def force_sync(
        self,
        model: "PeftModel",
        accelerator: "Accelerator",
        step: int,
    ) -> bool:
        """Force a weight sync regardless of step interval.

        Args:
            model: PEFT model with LoRA adapter
            accelerator: Accelerator instance for unwrapping model
            step: Current training step

        Returns:
            True if sync was successful, False otherwise
        """
        # Always save to "current" for vLLM sync
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(self._current_checkpoint))

        # Unload old adapter if one was previously loaded
        if self._adapter_loaded:
            if not self._unload_adapter():
                logger.warning("Failed to unload old LoRA adapter, continuing anyway")

        # Load new adapter from current checkpoint
        if not self._load_adapter(str(self._current_checkpoint)):
            logger.error("Failed to load LoRA adapter at step %d", step)
            return False

        self._last_sync_step = step
        self._adapter_loaded = True
        logger.info("Synced LoRA weights to vLLM at step %d", step)

        # Save permanent checkpoint every N steps
        if step % self._save_steps == 0 and step > self._last_save_step:
            adapter_path = self._checkpoint_dir / f"step_{step}"
            adapter_path.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(str(adapter_path))
            self._last_save_step = step
            logger.info("Saved permanent LoRA checkpoint to %s", adapter_path)

        return True

    def _load_adapter(self, adapter_path: str) -> bool:
        """Load LoRA adapter into vLLM server.

        Args:
            adapter_path: Path to the adapter checkpoint

        Returns:
            True if successful, False otherwise
        """
        if self._vllm_backend is not None:
            return self._vllm_backend.load_lora_adapter(
                lora_name=self._lora_name,
                lora_path=adapter_path,
            )

        payload = {
            "lora_name": self._lora_name,
            "lora_path": adapter_path,
        }

        try:
            response = self._client.post("/v1/load_lora_adapter", json=payload)
            if response.status_code != 200:
                logger.error(
                    "Failed to load LoRA adapter (status=%s): %s",
                    response.status_code,
                    response.text[:512],
                )
                return False
            return True
        except Exception as exc:
            logger.error("Error loading LoRA adapter: %s", exc)
            return False

    def _unload_adapter(self) -> bool:
        """Unload the current LoRA adapter from vLLM server.

        Returns:
            True if successful, False otherwise
        """
        if self._vllm_backend is not None:
            return self._vllm_backend.unload_lora_adapter(self._lora_name)

        payload = {"lora_name": self._lora_name}

        try:
            response = self._client.post("/v1/unload_lora_adapter", json=payload)
            if response.status_code != 200:
                logger.warning(
                    "Failed to unload LoRA adapter (status=%s): %s",
                    response.status_code,
                    response.text[:512],
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Error unloading LoRA adapter: %s", exc)
            return False

    def get_lora_name(self) -> str:
        """Get the LoRA adapter name used for vLLM requests.

        Returns:
            The LoRA name to use in requests, or None if no adapter loaded
        """
        if self._adapter_loaded:
            return self._lora_name
        return None

    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the last synced checkpoint.

        Returns:
            Path to the last checkpoint, or None if no sync performed
        """
        if self._last_sync_step < 0:
            return None
        return self._checkpoint_dir / f"step_{self._last_sync_step}"

    def close(self) -> None:
        """Close the synchronizer and release resources."""
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
