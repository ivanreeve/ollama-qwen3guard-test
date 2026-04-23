#!/usr/bin/env python3
"""T4-friendly head-only OPF finetuning.

This script keeps the OPF backbone frozen and trains only the output head.
It is intended for small benchmark-targeted adaptation runs where full-model
AdamW finetuning is not feasible on Colab T4 hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import tiktoken

from opf._api import resolve_checkpoint_path
from opf._common.constants import SCHEMA_VERSION
from opf._common.label_space import resolve_checkpoint_label_space
from opf._core.runtime import _load_checkpoint_config, _resolve_n_ctx
from opf._core.sequence_labeling import build_label_info
from opf._model.model import Transformer
from opf._model.weights import save_named_tensors
from opf._train.runner import (
    _batch_to_tensors,
    _build_epoch_batches,
    _build_windows,
    _collect_example_ids,
    _ensure_output_dir,
    _format_duration,
    _load_custom_label_space,
    _masked_token_loss_and_accuracy,
    _prepare_tokenized_examples,
    _rebuild_output_head_for_target_labels,
    _resolve_output_dtype,
    _split_train_validation,
    LoopStats,
    OutputHeadRemapStats,
)


OPF_DEFAULT_CHECKPOINT_DIR = Path.home() / ".opf" / "privacy_filter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OPF output head only. Intended for T4-class GPUs."
    )
    parser.add_argument("dataset", type=str, help="Training dataset JSONL path/glob")
    parser.add_argument(
        "--validation-dataset",
        type=str,
        default=None,
        help="Optional explicit validation dataset path/glob",
    )
    parser.add_argument(
        "--dataset-variant",
        choices=("full", "message"),
        default="full",
    )
    parser.add_argument(
        "--validation-dataset-variant",
        choices=("full", "message"),
        default=None,
    )
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--label-space-json", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--output-param-dtype",
        choices=("inherit", "bf16", "fp32"),
        default="inherit",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--summary-name",
        type=str,
        default="head_only_finetune_summary.json",
    )
    return parser.parse_args()


def has_complete_opf_checkpoint(checkpoint_dir: str | Path) -> bool:
    path = Path(checkpoint_dir).expanduser()
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def resolve_checkpoint_for_head_only(checkpoint_arg: str | None) -> str:
    if checkpoint_arg:
        return resolve_checkpoint_path(checkpoint_arg)

    cache_root = OPF_DEFAULT_CHECKPOINT_DIR
    original_dir = cache_root / "original"
    if has_complete_opf_checkpoint(original_dir):
        return str(original_dir)
    if has_complete_opf_checkpoint(cache_root):
        return str(cache_root)

    from huggingface_hub import snapshot_download

    print(
        "OPF checkpoint not found locally. "
        f"Downloading/resuming into {cache_root}...",
        flush=True,
    )
    snapshot_download(
        repo_id="openai/privacy-filter",
        local_dir=str(cache_root),
        allow_patterns=["original/*"],
    )
    if has_complete_opf_checkpoint(original_dir):
        return str(original_dir)
    if has_complete_opf_checkpoint(cache_root):
        return str(cache_root)
    raise RuntimeError(f"No usable OPF checkpoint found under {cache_root}")


def freeze_backbone(model: Transformer) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        trainable = name.startswith("unembedding.")
        param.requires_grad = trainable
        if not trainable:
            frozen += param.numel()
    model.eval()
    return frozen


def forward_head_only_logits(
    model: Transformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    with torch.no_grad():
        hidden = model.embedding(tokens)
        for block in model.block:
            hidden = block(hidden, attention_mask=attention_mask)
        hidden = model.norm(hidden)
    return F.linear(hidden.float(), model.unembedding.weight.float(), None)


def train_one_epoch_head_only(
    *,
    model: Transformer,
    windows: Sequence,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    rng: random.Random,
    pad_token_id: int,
    pad_label_id: int,
    epoch_index: int,
    num_epochs: int,
    expected_examples: int,
    progress_interval_s: float,
) -> LoopStats:
    epoch_batches = _build_epoch_batches(windows, batch_size=batch_size, rng=rng)
    if not epoch_batches:
        raise ValueError("No training batches were built")

    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_tokens = 0
    total_correct = 0
    optimizer_steps = 0
    processed_windows = 0
    seen_example_ids: set[str] = set()
    total_batches = len(epoch_batches)
    epoch_start = time.perf_counter()
    next_progress_at = (
        epoch_start + progress_interval_s if progress_interval_s > 0.0 else float("inf")
    )

    for batch_idx, batch_windows in enumerate(epoch_batches, start=1):
        processed_windows += len(batch_windows)
        seen_example_ids.update(_collect_example_ids(batch_windows))
        tokens, labels, masks = _batch_to_tensors(
            batch_windows,
            device=device,
            pad_token_id=pad_token_id,
            pad_label_id=pad_label_id,
        )
        logits = forward_head_only_logits(model, tokens, masks.bool())
        loss, batch_tokens, batch_correct = _masked_token_loss_and_accuracy(
            logits=logits,
            labels=labels,
            masks=masks,
        )
        if batch_tokens == 0:
            continue

        total_loss_sum += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens
        total_correct += batch_correct

        (loss / grad_accum_steps).backward()
        should_step = batch_idx % grad_accum_steps == 0 or batch_idx == total_batches
        if not should_step:
            continue

        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([model.unembedding.weight], max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

        now = time.perf_counter()
        if now >= next_progress_at or batch_idx == total_batches:
            elapsed = now - epoch_start
            batch_fraction = float(batch_idx) / float(total_batches)
            estimated_epoch_total = elapsed / batch_fraction if batch_fraction > 0.0 else 0.0
            eta_epoch_s = max(0.0, estimated_epoch_total - elapsed)
            remaining_epochs = max(0, num_epochs - epoch_index)
            eta_total_s = eta_epoch_s + (estimated_epoch_total * remaining_epochs)
            running_loss = total_loss_sum / total_tokens if total_tokens > 0 else 0.0
            running_acc = float(total_correct) / float(total_tokens) if total_tokens > 0 else 0.0
            print(
                "head-only train progress: "
                f"epoch={epoch_index}/{num_epochs} "
                f"batch={batch_idx}/{total_batches} "
                f"windows={processed_windows}/{len(windows)} "
                f"examples_seen={len(seen_example_ids)}/{expected_examples} "
                f"tokens={total_tokens} "
                f"train_loss={running_loss:.6f} "
                f"train_token_accuracy={running_acc:.4f} "
                f"eta_epoch={_format_duration(eta_epoch_s)} "
                f"eta_total={_format_duration(eta_total_s)}",
                flush=True,
            )
            if progress_interval_s > 0.0:
                next_progress_at = now + progress_interval_s

    if total_tokens <= 0:
        raise ValueError("No valid training tokens were observed")

    return LoopStats(
        loss=total_loss_sum / total_tokens,
        token_accuracy=float(total_correct) / float(total_tokens),
        tokens=total_tokens,
        batches=len(epoch_batches),
        optimizer_steps=optimizer_steps,
    )


def evaluate_head_only(
    *,
    model: Transformer,
    windows: Sequence,
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
    pad_label_id: int,
) -> LoopStats:
    batches = _build_epoch_batches(windows, batch_size=batch_size, rng=random.Random(0))
    if not batches:
        raise ValueError("No validation batches were built")

    total_loss_sum = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.inference_mode():
        for batch_windows in batches:
            tokens, labels, masks = _batch_to_tensors(
                batch_windows,
                device=device,
                pad_token_id=pad_token_id,
                pad_label_id=pad_label_id,
            )
            logits = forward_head_only_logits(model, tokens, masks.bool())
            loss, batch_tokens, batch_correct = _masked_token_loss_and_accuracy(
                logits=logits,
                labels=labels,
                masks=masks,
            )
            if batch_tokens == 0:
                continue
            total_loss_sum += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
            total_correct += batch_correct

    if total_tokens <= 0:
        raise ValueError("No valid validation tokens were observed")

    return LoopStats(
        loss=total_loss_sum / total_tokens,
        token_accuracy=float(total_correct) / float(total_tokens),
        tokens=total_tokens,
        batches=len(batches),
    )


def main() -> int:
    args = parse_args()

    progress_interval_s = 15.0
    os.environ.setdefault("OPF_MOE_TRITON", "0")

    checkpoint = resolve_checkpoint_for_head_only(args.checkpoint)
    device = torch.device(args.device)
    base_config = _load_checkpoint_config(checkpoint)
    n_ctx = _resolve_n_ctx(base_config, args.n_ctx, device)

    encoding_name = base_config.get("encoding")
    if not isinstance(encoding_name, str) or not encoding_name:
        raise ValueError("Checkpoint config field encoding must be a non-empty string")
    encoding = tiktoken.get_encoding(encoding_name)
    pad_token_id = int(encoding.eot_token)

    (
        checkpoint_category_version,
        _checkpoint_span_class_names,
        checkpoint_ner_class_names,
    ) = resolve_checkpoint_label_space(checkpoint)

    custom_label_space = _load_custom_label_space(args.label_space_json)
    if custom_label_space is None:
        resolved_category_version = checkpoint_category_version
        resolved_ner_class_names = checkpoint_ner_class_names
        label_space_source = "checkpoint"
        resolved_label_space_path = None
    else:
        (
            resolved_category_version,
            _custom_span_class_names,
            resolved_ner_class_names,
            resolved_label_space_path,
        ) = custom_label_space
        label_space_source = "label-space-json"

    label_info = build_label_info(resolved_ner_class_names)
    base_ner_class_names = checkpoint_ner_class_names

    train_examples_all = _prepare_tokenized_examples(
        dataset_path=args.dataset,
        dataset_variant=args.dataset_variant,
        encoding=encoding,
        label_info=label_info,
        max_examples=args.max_train_examples,
    )
    if not train_examples_all:
        raise ValueError("No training examples were loaded")

    validation_variant = args.validation_dataset_variant or args.dataset_variant
    if args.validation_dataset:
        train_examples = train_examples_all
        validation_examples = _prepare_tokenized_examples(
            dataset_path=args.validation_dataset,
            dataset_variant=validation_variant,
            encoding=encoding,
            label_info=label_info,
            max_examples=args.max_validation_examples,
        )
    else:
        train_examples, validation_examples = _split_train_validation(
            train_examples_all,
            validation_split=args.validation_split,
            shuffle_seed=args.shuffle_seed,
        )
        if args.max_validation_examples is not None:
            validation_examples = validation_examples[: args.max_validation_examples]

    train_windows = _build_windows(train_examples, n_ctx=n_ctx)
    validation_windows = _build_windows(validation_examples, n_ctx=n_ctx)
    if not train_windows:
        raise ValueError("No train windows were produced")

    model = Transformer.from_checkpoint(checkpoint, device=device)
    model = model.to(dtype=torch.float32)
    output_head_remap_stats = OutputHeadRemapStats(exact_rows_copied=0, fallback_rows_copied=0)
    output_head_reinitialized = False
    if tuple(resolved_ner_class_names) != tuple(base_ner_class_names):
        output_head_remap_stats = _rebuild_output_head_for_target_labels(
            model,
            base_ner_class_names=base_ner_class_names,
            target_ner_class_names=resolved_ner_class_names,
            device=device,
        )
        output_head_reinitialized = True

    frozen_params = freeze_backbone(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_param_count = sum(param.numel() for param in trainable_params)

    print(
        "head-only training plan: "
        f"epochs={args.epochs} "
        f"train_examples={len(train_examples)} "
        f"train_windows={len(train_windows)} "
        f"validation_examples={len(validation_examples)} "
        f"validation_windows={len(validation_windows)} "
        f"trainable_params={trainable_param_count} "
        f"frozen_params={frozen_params} "
        f"resolved_n_ctx={n_ctx}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    epoch_rng = random.Random(args.shuffle_seed)
    epoch_summaries: list[dict[str, object]] = []
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metric_name = "validation_loss" if validation_windows else "train_loss"
    overall_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch_head_only(
            model=model,
            windows=train_windows,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            rng=epoch_rng,
            pad_token_id=pad_token_id,
            pad_label_id=label_info.background_token_label,
            epoch_index=epoch,
            num_epochs=args.epochs,
            expected_examples=len(train_examples),
            progress_interval_s=progress_interval_s,
        )

        validation_stats: LoopStats | None = None
        if validation_windows:
            validation_stats = evaluate_head_only(
                model=model,
                windows=validation_windows,
                device=device,
                batch_size=args.batch_size,
                pad_token_id=pad_token_id,
                pad_label_id=label_info.background_token_label,
            )

        tracked_metric = validation_stats.loss if validation_stats is not None else train_stats.loss
        if tracked_metric < best_metric:
            best_metric = tracked_metric
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        epoch_summary: dict[str, object] = {
            "epoch": epoch,
            "elapsed_s": time.perf_counter() - epoch_start,
            "train_loss": train_stats.loss,
            "train_token_accuracy": train_stats.token_accuracy,
            "train_tokens": train_stats.tokens,
            "train_batches": train_stats.batches,
            "optimizer_steps": train_stats.optimizer_steps,
        }
        if validation_stats is not None:
            epoch_summary.update(
                {
                    "validation_loss": validation_stats.loss,
                    "validation_token_accuracy": validation_stats.token_accuracy,
                    "validation_tokens": validation_stats.tokens,
                    "validation_batches": validation_stats.batches,
                }
            )
        epoch_summaries.append(epoch_summary)

        if validation_stats is None:
            print(
                f"epoch {epoch}/{args.epochs}: train_loss={train_stats.loss:.6f} "
                f"train_token_accuracy={train_stats.token_accuracy:.4f} "
                f"optimizer_steps={train_stats.optimizer_steps}",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch}/{args.epochs}: train_loss={train_stats.loss:.6f} "
                f"val_loss={validation_stats.loss:.6f} "
                f"val_token_accuracy={validation_stats.token_accuracy:.4f} "
                f"optimizer_steps={train_stats.optimizer_steps}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Training finished without any tracked metrics")
    model.load_state_dict(best_state, strict=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_output_dir(output_dir, overwrite=args.overwrite_output)
    serialized_param_dtype, output_dtype = _resolve_output_dtype(
        output_param_dtype_flag=args.output_param_dtype,
        base_config=base_config,
    )

    output_config = dict(base_config)
    output_config["param_dtype"] = serialized_param_dtype
    output_config["num_labels"] = len(resolved_ner_class_names)
    output_config["category_version"] = resolved_category_version
    output_config["span_class_names"] = list(label_info.span_class_names)
    output_config["ner_class_names"] = list(resolved_ner_class_names)
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    weights_path = output_dir / "model.safetensors"
    save_named_tensors(
        weights_path,
        {name: param for name, param in model.named_parameters()},
        dtype=output_dtype,
    )

    summary_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "mode": "head_only",
        "base_checkpoint": checkpoint,
        "output_checkpoint_dir": str(output_dir),
        "checkpoint_category_version": checkpoint_category_version,
        "resolved_category_version": resolved_category_version,
        "label_space_source": label_space_source,
        "label_space_json_path": resolved_label_space_path,
        "num_output_labels": len(resolved_ner_class_names),
        "output_head_reinitialized": output_head_reinitialized,
        "output_head_rows_copied": output_head_remap_stats.total_rows_copied,
        "output_head_rows_copied_exact": output_head_remap_stats.exact_rows_copied,
        "output_head_rows_copied_fallback": output_head_remap_stats.fallback_rows_copied,
        "trainable_params": trainable_param_count,
        "frozen_params": frozen_params,
        "span_class_names": list(label_info.span_class_names),
        "encoding": encoding_name,
        "resolved_n_ctx": n_ctx,
        "device": str(device),
        "train_dataset": args.dataset,
        "train_dataset_variant": args.dataset_variant,
        "validation_dataset": args.validation_dataset,
        "validation_dataset_variant": validation_variant if args.validation_dataset else None,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "best_epoch": best_epoch,
        "epoch_summaries": epoch_summaries,
        "elapsed_s": time.perf_counter() - overall_start,
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    usage_lines = [
        "Head-only OPF finetuned checkpoint",
        "",
        f"Base checkpoint: {checkpoint}",
        f"Mode: head_only",
        f"Label space source: {label_space_source}",
        f"Resolved category version: {resolved_category_version}",
        f"Output labels: {len(resolved_ner_class_names)}",
        f"Trainable params: {trainable_param_count}",
        f"Frozen params: {frozen_params}",
        "",
        "To evaluate with this repo:",
        f"  python detect_pii.py --model {output_dir}",
    ]
    (output_dir / "USAGE.txt").write_text("\n".join(usage_lines) + "\n", encoding="utf-8")

    print(f"Saved checkpoint to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
