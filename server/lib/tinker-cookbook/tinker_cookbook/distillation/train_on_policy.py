"""
Implements on-policy distillation. For more details, see:
https://thinkingmachines.ai/blog/on-policy-distillation
"""

import asyncio
import logging
import os
import time
from typing import Any, List, Literal, Sequence, Dict, cast

import chz
import tinker
import torch
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import scope, get_scope_context, trace_init

# Dataset configuration classes
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
)

# We re-use these methods from the RL training recipe
from tinker_cookbook.rl.train import (
    save_checkpoint_and_get_sampling_client,
    train_step,
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
)

logger = logging.getLogger(__name__)

def _get_or_create_teacher_tokenizer(
    cache: dict[str, Tokenizer],
    *,
    model_name: str,
) -> Tokenizer:
    if model_name not in cache:
        cache[model_name] = get_tokenizer(model_name)
    return cache[model_name]

def _teacher_input_and_completion_start(
    *,
    raw_prompt_text: str,
    student_completion_text: str,
    convo_prefix: list[renderers.Message] | None,
    teacher_renderer: renderers.Renderer,
    teacher_tokenizer: Tokenizer,
) -> tuple[tinker.ModelInput, int]:
    messages = list(convo_prefix or []) + [
        {"role": "user", "content": raw_prompt_text},
    ]
    prompt_input = teacher_renderer.build_generation_prompt(messages)
    prompt_tokens = list(prompt_input.to_ints())
    completion_tokens = teacher_tokenizer.encode(
        student_completion_text,
        add_special_tokens=False,
    )

    prompt_snippet = teacher_tokenizer.decode(
        prompt_tokens[: min(len(prompt_tokens), 20)],
        skip_special_tokens=False,
    )
    completion_snippet = student_completion_text[:40].replace("\n", " ")
    logger.info(
        "GOLD: teacher_prompt_prefix='%s...' student_completion_prefix='%s...'",
        prompt_snippet,
        completion_snippet,
    )
    return (
        tinker.ModelInput.from_ints(tokens=prompt_tokens + completion_tokens),
        len(prompt_tokens),
    )

@scope
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    teacher_clients_D: List[tinker.SamplingClient],
    dataset_indices_D: List[int],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    *,
    student_sequence_token_ids: List[List[int]] | None = None,
    student_completion_texts: List[str] | None = None,
    student_full_texts: List[str] | None = None,
    student_prompt_texts: List[str] | None = None,
    student_completion_start_indices: List[int] | None = None,
    raw_prompt_texts: List[str] | None = None,
    teacher_convo_prefixes: List[list[renderers.Message]] | None = None,
    student_tokenizer: Tokenizer | None = None,
    teacher_tokenizers: List[Tokenizer] | None = None,
    teacher_renderers: List[renderers.Renderer] | None = None,
    use_gold_alignment: bool = False,
) -> Dict[str, float]:
    """
    Compute reverse KL between the student (log p) and the teacher model (log q), computed as
    log p - log q. We then adjust the advantages in-place as the negative reverse KL.

    Args:
        data_D: List of datums to compute KL for
        teacher_clients_D: List of teacher sampling clients, one per datum
        dataset_indices_D: List of dataset indices, one per datum
        kl_penalty_coef: Coefficient for KL penalty
        kl_discount_factor: Discount factor for future KL
    """
    # Note: if your teacher has a different renderer than the student, you may want to modify
    #       the full_sequence_inputs_D to match the teacher's renderer.
    teacher_completion_start_offsets = None
    if use_gold_alignment:
        if (
            student_full_texts is None
            or student_completion_texts is None
            or student_sequence_token_ids is None
            or student_prompt_texts is None
            or student_completion_start_indices is None
            or raw_prompt_texts is None
            or teacher_convo_prefixes is None
            or student_tokenizer is None
            or teacher_tokenizers is None
            or teacher_renderers is None
        ):
            raise ValueError("[tinker-cookbook.distillation.train_on_policy] GOLD alignment requires reconstructed student text and teacher tokenizers")

        teacher_inputs_and_offsets = [
            _teacher_input_and_completion_start(
                raw_prompt_text=raw_prompt_texts[i],
                student_completion_text=student_completion_texts[i],
                convo_prefix=teacher_convo_prefixes[i],
                teacher_renderer=teacher_renderers[dataset_indices_D[i]],
                teacher_tokenizer=teacher_tokenizers[dataset_indices_D[i]],
            )
            for i in range(len(data_D))
        ]
        full_sequence_inputs_D = [entry[0] for entry in teacher_inputs_and_offsets]
        teacher_completion_start_offsets = [entry[1] for entry in teacher_inputs_and_offsets]
        logger.info(
            "GOLD: encoded teacher inputs for logprobs with teacher template"
        )
    else:
        full_sequence_inputs_D = [
            datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
            for datum in data_D
        ]
    # Compute the teacher's logprobs for each element of the batch
    # Each datum uses its corresponding teacher sampling client
    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(sequence_input)
            for teacher_client, sequence_input in zip(teacher_clients_D, full_sequence_inputs_D)
        ]
    )
    # The reverse KL is computed as KL[p||q] = log p - log q, where
    #   - p: sampled_logprobs
    #   - q: teacher_logprobs
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    reverse_kl = []
    effective_masks = []

    for i, (teacher_logprobs, sampled_logprobs, mask) in enumerate(
        safezip(teacher_logprobs_D, sampled_logprobs_D, float_masks)
    ):
        teacher_tensor = torch.tensor(teacher_logprobs[1:], dtype=sampled_logprobs.dtype)
        mask_tensor = mask

        if use_gold_alignment:
            teacher_tokenizer = teacher_tokenizers[dataset_indices_D[i]]
            student_tokens_full = student_sequence_token_ids[i]
            student_start = student_completion_start_indices[i] if student_completion_start_indices else 0
            completion_start = teacher_completion_start_offsets[i] if teacher_completion_start_offsets else 0
            if completion_start >= teacher_tensor.shape[0]:
                logger.warning(
                    "GOLD: teacher completion start %d exceeds teacher tensor len %d; skipping trim",
                    completion_start,
                    int(teacher_tensor.shape[0]),
                )
                completion_start = int(teacher_tensor.shape[0])
            original_teacher_len = int(teacher_tensor.shape[0])
            if completion_start:
                teacher_tensor = teacher_tensor[completion_start:]

            if student_start >= len(student_tokens_full):
                logger.warning(
                    "GOLD: student completion start %d exceeds token len %d; clamping",
                    student_start,
                    len(student_tokens_full),
                )
                student_start = len(student_tokens_full)
            student_tokens = student_tokens_full[student_start:]
            student_logprobs = sampled_logprobs[student_start:]
            student_mask = mask_tensor[student_start:]
            teacher_completion_ids = _student_completion_to_teacher_tokens(
                student_completion_texts[i],
                teacher_tokenizer,
            )
            logger.info(
                "GOLD: datum=%d trimmed teacher_completion_tokens=%d/%d student_completion_tokens=%d/%d",
                i,
                int(teacher_tensor.shape[0]),
                original_teacher_len,
                len(student_tokens),
                len(student_tokens_full),
            )

            seq_len = min(len(teacher_completion_ids), int(teacher_tensor.shape[0]))
            if seq_len != len(teacher_completion_ids):
                teacher_completion_ids = teacher_completion_ids[:seq_len]
                teacher_tensor = teacher_tensor[:seq_len]

            assert len(teacher_completion_ids) == int(
                teacher_tensor.shape[0]
            ), "GOLD: teacher token/logprobs lengths diverged after trimming"
            assert len(student_tokens) == student_logprobs.shape[0], (
                "GOLD: student completion slices must have matching lengths"
            )

            group_reverse_kl_slice, group_mask_slice = _compute_groupwise_reverse_kl(
                student_tokenizer,
                student_tokens,
                student_logprobs,
                teacher_tokenizer,
                teacher_completion_ids,
                teacher_tensor,
                student_mask,
            )
            if student_start:
                padded_reverse_kl = torch.zeros_like(sampled_logprobs)
                padded_mask = torch.zeros_like(mask_tensor)
                padded_reverse_kl[student_start:] = group_reverse_kl_slice
                padded_mask[student_start:] = group_mask_slice
            else:
                padded_reverse_kl = group_reverse_kl_slice
                padded_mask = group_mask_slice
            reverse_kl.append(padded_reverse_kl)
            effective_masks.append(padded_mask)
            continue

        reverse_kl.append((sampled_logprobs - teacher_tensor) * mask_tensor)
        effective_masks.append(mask_tensor)

    # Track per-dataset KL for logging
    # dataset_idx -> (sum of KL, sum of mask)
    per_dataset_kl: Dict[int, tuple[float, float]] = {}

    for i, datum in enumerate(data_D):
        # The advantage is the negative reverse KL. We can optionally apply a discount factor.
        kl_advantages = -kl_penalty_coef * effective_masks[i] * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

        # Accumulate per-dataset KL
        dataset_idx = dataset_indices_D[i]
        kl_sum = reverse_kl[i].sum().item()
        mask_sum = effective_masks[i].sum().item()
        if dataset_idx not in per_dataset_kl:
            per_dataset_kl[dataset_idx] = (0.0, 0.0)
        prev_kl_sum, prev_mask_sum = per_dataset_kl[dataset_idx]
        per_dataset_kl[dataset_idx] = (prev_kl_sum + kl_sum, prev_mask_sum + mask_sum)

    # Compute average reverse KL over the batch for logging purposes
    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / sum(
        [mask.sum() for mask in effective_masks]
    )

    # Compute per-dataset metrics
    metrics = {"teacher_kl": float(avg_logp_diff)}
    for dataset_idx, (kl_sum, mask_sum) in per_dataset_kl.items():
        if mask_sum > 0:
            metrics[f"teacher_kl/dataset_{dataset_idx}"] = float(kl_sum / mask_sum)

    return metrics

def _datum_to_student_completion_ids(datum: tinker.Datum) -> List[int]:
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    return [int(token) for token in target_tokens]

def _student_text_to_teacher_input(*, text: str, tokenizer: Tokenizer) -> tinker.ModelInput:
    teacher_token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    return tinker.ModelInput.from_ints(tokens=teacher_token_ids)

def _student_completion_to_teacher_tokens(text: str, tokenizer: Tokenizer) -> List[int]:
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )

def _token_pieces(tokenizer: Tokenizer, token_ids: List[int]) -> List[str]:
    pieces = []
    prev = ""
    for idx in range(len(token_ids)):
        cur = tokenizer.decode(
            token_ids[: idx + 1],
            skip_special_tokens=False,
        )
        pieces.append(cur[len(prev):])
        prev = cur
    return pieces

def _compute_groupwise_reverse_kl(
    student_tokenizer: Tokenizer,
    student_token_ids: List[int],
    student_logprobs: torch.Tensor,
    teacher_tokenizer: Tokenizer,
    teacher_token_ids: List[int],
    teacher_logprobs: torch.Tensor,
    base_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_groups, teacher_groups = _build_alignment_groups(
        student_tokenizer,
        student_token_ids,
        teacher_tokenizer,
        teacher_token_ids,
    )
    logger.info(
        "GOLD alignment (post-trim): student_tokens=%d teacher_tokens=%d student_groups=%d teacher_groups=%d",
        len(student_token_ids),
        len(teacher_token_ids),
        len(student_groups),
        len(teacher_groups)
    )

    reverse_kl = torch.zeros_like(student_logprobs)
    mask = torch.zeros_like(base_mask)

    for s_group, t_group in zip(student_groups, teacher_groups):
        teacher_indices = [idx for idx in t_group if idx < len(teacher_logprobs)]
        student_indices = [idx for idx in s_group if idx < len(student_logprobs) and base_mask[idx] > 0]
        if not teacher_indices or not student_indices:
            continue

        student_log_sum = student_logprobs[student_indices[0]]
        for s_idx in student_indices[1:]:
            student_log_sum = student_log_sum + student_logprobs[s_idx]
            
        teacher_log_sum = teacher_logprobs[teacher_indices[0]]
        for t_idx in teacher_indices[1:]:
            teacher_log_sum = teacher_log_sum + teacher_logprobs[t_idx]

        delta = student_log_sum - teacher_log_sum
        share = delta / len(student_indices)

        for s_idx in student_indices:
            reverse_kl[s_idx] = share
            mask[s_idx] = base_mask[s_idx]

    return reverse_kl, mask

def _build_alignment_groups(
    student_tokenizer: Tokenizer,
    student_token_ids: List[int],
    teacher_tokenizer: Tokenizer,
    teacher_token_ids: List[int],
) -> tuple[List[List[int]], List[List[int]]]:
    student_pieces = _token_pieces(student_tokenizer, student_token_ids)
    teacher_pieces = _token_pieces(teacher_tokenizer, teacher_token_ids)

    student_groups = []
    teacher_groups = []

    i = j = 0
    s_buf = t_buf = ""
    cur_s = []
    cur_t = []

    def flush() -> None:
        nonlocal s_buf, t_buf, cur_s, cur_t
        if cur_s and cur_t:
            student_groups.append(cur_s.copy())
            teacher_groups.append(cur_t.copy())
        s_buf = t_buf = ""
        cur_s = []
        cur_t = []

    while i < len(student_pieces) or j < len(teacher_pieces):
        if s_buf == t_buf and s_buf != "":
            flush()
            continue
        if i < len(student_pieces) and (s_buf == "" or len(s_buf) <= len(t_buf)):
            s_buf += student_pieces[i]
            cur_s.append(i)
            i += 1
            continue
        if j < len(teacher_pieces):
            t_buf += teacher_pieces[j]
            cur_t.append(j)
            j += 1
            continue
        break

    if s_buf == t_buf and s_buf != "":
        flush()

    return student_groups, teacher_groups

@chz.chz
class Config:
    learning_rate: float
    dataset_configs: List[DistillationDatasetConfig]
    model_name: str
    max_tokens: int
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None

    use_gold_alignment: bool = False


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    *,
    teacher_tokenizers: List[Tokenizer] | None = None,
    teacher_renderers: List[renderers.Renderer] | None = None,
    use_gold_alignment: bool = False,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Print one datum per dataset
    printed_datasets = set()
    for datum, metadata in zip(data_D, metadata_D):
        dataset_idx = dataset_indices_P[metadata["group_idx"]]
        if dataset_idx not in printed_datasets:
            logger.info(colorize_example(datum, tokenizer, key="mask"))
            printed_datasets.add(dataset_idx)

    student_sequence_token_ids = None
    student_completion_texts = None
    student_full_texts = None
    student_prompt_texts = None
    student_completion_start_indices = None
    raw_prompt_texts = None
    teacher_convo_prefixes = None
    if use_gold_alignment:
        student_sequence_token_ids = []
        student_completion_texts = []
        student_full_texts = []
        student_prompt_texts = []
        student_completion_start_indices = []
        raw_prompt_texts = []
        teacher_convo_prefixes = []
        group_prompts = []
        group_convo_prefixes = []
        for env_group_builder in env_group_builders_P:
            metadata = getattr(env_group_builder, "metadata", None) or {}
            group_prompts.append(metadata.get("prompt"))
            group_convo_prefixes.append(metadata.get("convo_prefix"))
        for datum in data_D:
            prompt_ids = list(datum.model_input.to_ints())
            completion_ids = _datum_to_student_completion_ids(datum)
            student_sequence_token_ids.append(prompt_ids + completion_ids)
            student_completion_texts.append(
                tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=False,
                )
            )
            student_prompt_texts.append(
                tokenizer.decode(prompt_ids, skip_special_tokens=False)
            )
            student_completion_start_indices.append(len(prompt_ids))
            full_ids = prompt_ids + completion_ids
            student_full_texts.append(
                tokenizer.decode(full_ids, skip_special_tokens=False)
            )
        for metadata in metadata_D:
            group_idx = metadata["group_idx"]
            raw_prompt = group_prompts[group_idx]
            if raw_prompt is None:
                raise ValueError("GOLD alignment requires prompt metadata per group")
            raw_prompt_texts.append(raw_prompt)
            teacher_convo_prefixes.append(group_convo_prefixes[group_idx])
            
        logger.info(
            "GOLD: reconstructed student text (prompts=%d)",
            len(student_full_texts)
        )

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("compute_kl_penalty", metrics):
            # Map each datum to its teacher sampling client and dataset index using metadata
            #   - metadata_D contains group_idx which indexes into trajectory_groups_P
            #   - dataset_indices_P[group_idx] gives us the dataset index
            #   - teacher_clients[dataset_idx] gives us the teacher
            teacher_clients_D = [
                teacher_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
            ]
            dataset_indices_D = [
                dataset_indices_P[metadata["group_idx"]] for metadata in metadata_D
            ]
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                teacher_clients_D,
                dataset_indices_D,
                kl_penalty_coef,
                kl_discount_factor,
                student_sequence_token_ids=student_sequence_token_ids,
                student_completion_texts=student_completion_texts,
                student_full_texts=student_full_texts,
                student_prompt_texts=student_prompt_texts,
                student_completion_start_indices=student_completion_start_indices,
                raw_prompt_texts=raw_prompt_texts,
                teacher_convo_prefixes=teacher_convo_prefixes,
                student_tokenizer=tokenizer,
                teacher_tokenizers=teacher_tokenizers,
                teacher_renderers=teacher_renderers,
                use_gold_alignment=use_gold_alignment,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    teacher_tokenizers: List[Tokenizer],
    teacher_renderers: List[renderers.Renderer],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    context = get_scope_context()
    context.attributes["step"] = i_batch

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        dataset_indices_P,
        teacher_clients,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        teacher_tokenizers=teacher_tokenizers,
        teacher_renderers=teacher_renderers,
        use_gold_alignment=cfg.use_gold_alignment,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: CompositeDataset,
    teacher_clients: List[tinker.SamplingClient],
    teacher_tokenizers: List[Tokenizer],
    teacher_renderers: List[renderers.Renderer],
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P, dataset_indices_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=False,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            dataset_indices_P,
            teacher_clients,
            teacher_tokenizers,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for on-policy distillation."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create datasets and teacher sampling clients from configs
    datasets = []
    teacher_clients = []
    teacher_tokenizers = []
    teacher_renderers = []
    teacher_tokenizer_cache = {}
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]

    for dataset_config in cfg.dataset_configs:
        # Create dataset
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)

        # Add test dataset evaluator if present
        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

        # Create teacher sampling client
        teacher_config = dataset_config.teacher_config
        teacher_client = service_client.create_sampling_client(base_model=teacher_config.base_model)
        # Load teacher checkpoint if specified
        if teacher_config.load_checkpoint_path is not None:
            teacher_client = service_client.create_sampling_client(
                base_model=teacher_config.base_model,
                model_path=teacher_config.load_checkpoint_path,
            )
        teacher_clients.append(teacher_client)
        teacher_tokenizer =_get_or_create_teacher_tokenizer(
            teacher_tokenizer_cache,
            model_name=teacher_config.base_model,
        )
        teacher_tokenizers.append(teacher_tokenizer)
        renderer_name = model_info.get_recommended_renderer_name(teacher_config.base_model)
        teacher_renderers.append(
            renderers.get_renderer(renderer_name, tokenizer=teacher_tokenizer)
        )

        logger.info(
            f"Created teacher sampling client for {teacher_config.base_model} "
            f"(checkpoint: {teacher_config.load_checkpoint_path})"
        )

    # Wrap datasets in CompositeDataset
    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,
        teacher_tokenizers=teacher_tokenizers,
        teacher_renderers=teacher_renderers,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
