"""
Train on the sampled trajectories from Qwen3-405B.

This script:
1. Loads the checkpoint from main.py training

"""

import os
import dotenv
import tinker
import datasets
import logging
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
import time
import tqdm
import chz
import json
dotenv.load_dotenv()


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def load_checkpoint_path(log_path: str, checkpoint_name: str) -> str | None:
    """
    Load checkpoint path from checkpoints.jsonl file.
    Returns the Tinker URI for the checkpoint, or None if not found.
    """
    checkpoints_file = os.path.join(log_path, "checkpoints.jsonl")
    if not os.path.exists(checkpoints_file):
        logger.warning(f"Checkpoints file not found: {checkpoints_file}")
        return None
    
    # Read all checkpoints and find the requested one
    with open(checkpoints_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            checkpoint = json.loads(line)
            if checkpoint.get("name") == checkpoint_name:
                state_path = checkpoint.get("state_path")
                logger.info(f"Found checkpoint '{checkpoint_name}' at batch {checkpoint.get('batch')}")
                return state_path
    
    logger.warning(f"Checkpoint '{checkpoint_name}' not found in {checkpoints_file}")
    return None


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "./logs_sampled/"
    previous_log_path: str = "./logs/"  # Path to logs from main.py training
    previous_checkpoint_name: str = "swe-rebench-sft-70"  # Which checkpoint to load
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Model being trained
    hf_dataset_name: str = "nebius/SWE-rebench-openhands-trajectories"
    hf_split_name: str = "train"  # Split with sampled trajectories
    
    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 5e-6  # Lower LR for distillation/continued training
    warmup_steps: int = 50  # Warmup for stability
    max_length: int = 8192
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    lora_rank: int = 32
    save_every: int = 10
    weight_decay: float = 0.01  # Regularization


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load the sampled trajectories split
    logger.info(f"Loading '{config.hf_split_name}' split from {config.hf_dataset_name}...")
    ds = datasets.load_dataset(config.hf_dataset_name)
    assert isinstance(ds, datasets.DatasetDict)
    
    if config.hf_split_name not in ds:
        raise ValueError(
            f"Split '{config.hf_split_name}' not found in dataset. "
            f"Available splits: {list(ds.keys())}. "
            "Did you run sample_trajectories.py first?"
        )
    
    train_dataset = ds[config.hf_split_name]
    # Filtering relevant trajectories - resolved and also from the filtered set of repos
    train_dataset = train_dataset.filter(lambda x: x['resolved'] == True)
    split_dataset = datasets.load_dataset("collinear-ai/SWE-rebench-split")
    filtered_instance_ids_set = set(split_dataset['filtered']['instance_id'])
    train_dataset = train_dataset.filter(lambda x: x['instance_id'] in filtered_instance_ids_set)
    logger.info(f"Training on '{config.hf_split_name}' split with {len(train_dataset)} examples")

    n_train_batches = len(train_dataset) // config.batch_size
    logger.info(f"Train batches: {n_train_batches}")

    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Check for resume from sampled training checkpoints first
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        logger.info(f"Resuming sampled training from batch {resume_info['batch']}")
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
    else:
        # Load from the main.py checkpoint
        previous_checkpoint_path = load_checkpoint_path(
            config.previous_log_path, 
            config.previous_checkpoint_name
        )
        
        if previous_checkpoint_path:
            logger.info(f"Loading checkpoint from main.py training: {previous_checkpoint_path}")
            training_client = service_client.create_training_client_from_state_with_optimizer(
                previous_checkpoint_path
            )
            start_batch = 0
        else:
            logger.warning(
                f"No checkpoint '{config.previous_checkpoint_name}' found in {config.previous_log_path}, "
                "starting from scratch"
            )
            training_client = service_client.create_lora_training_client(
                base_model=config.model_name, rank=config.lora_rank
            )
            start_batch = 0

    logger.info(f"Training for {n_train_batches} steps")
    logger.info(f"Learning rate: {config.learning_rate} (with warmup over {config.warmup_steps} steps)")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Weight decay: {config.weight_decay}")
    logger.info(f"Training on distilled trajectories from Qwen3-235B")

    # Shuffle with consistent seed
    train_dataset = train_dataset.shuffle(seed=42)
    
    for batch_idx in tqdm.tqdm(range(start_batch, n_train_batches)):
        start_time = time.time()
        step = batch_idx
        metrics = {}

        # Save checkpoint
        if config.save_every > 0 and step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Learning rate schedule with warmup
        if step < config.warmup_steps:
            lr_mult = (step + 1) / config.warmup_steps
        else:
            remaining_steps = n_train_batches - config.warmup_steps
            current_step_after_warmup = step - config.warmup_steps
            lr_mult = max(0.1, 1.0 - current_step_after_warmup / remaining_steps)
        
        current_lr = config.learning_rate * lr_mult

        # Adam parameters with weight decay
        adam_params = tinker.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=config.weight_decay
        )

        # Get training batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        batch = [
            conversation_to_datum(
                row["trajectory"],  # type: ignore
                renderer,
                config.max_length,
                config.train_on_what,
            )
            for row in batch_rows
        ]

        # Training step
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Compute train metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        # Log metrics
        metrics.update(
            num_sequences=len(batch),
            num_tokens=sum(d.model_input.length for d in batch),
            learning_rate=current_lr,
            lr_multiplier=lr_mult,
            train_mean_nll=train_nll,
            progress=step / n_train_batches,
            time_total=time.time() - start_time,
            is_warmup=step < config.warmup_steps,
        )
        ml_logger.log_metrics(metrics=metrics, step=step)

        # Log every 10 steps
        if step % 10 == 0:
            logger.info(
                f"Step {step}/{n_train_batches} | "
                f"LR: {current_lr:.2e} | "
                f"Train NLL: {train_nll:.4f} | "
                f"Time: {metrics['time_total']:.2f}s"
            )

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )

    logger.info("Distillation training completed!")
    logger.info(f"Final checkpoint saved to {config.log_path}")
    ml_logger.close()


if __name__ == "__main__":
    main(Config())
