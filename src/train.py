import argparse
import math
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

import src.model  # noqa: F401 - registers model classes in ModelFactory.
from src.config import OptimizerConfig, SchedulerConfig, TrainConfig, ModelFactory
from src.data import PackedPretrainDataset, collate_fn


DEFAULT_PROMPTS = [
    "한국어 인공지능 모델을 직접 학습하는 이유는",
    "서울의 대중교통은",
    "과학 연구에서 데이터 품질이 중요한 까닭은",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MyGPT")
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        dest="dataset_path",
        type=Path,
        required=True,
        help="Dataset version root, e.g. .../datasets/pretrain/v1",
    )
    parser.add_argument(
        "--weights-dir",
        "--weights_dir",
        dest="weights_dir",
        type=Path,
        required=True,
        help="Directory where run checkpoints are written",
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--wandb", default="MyGPT", help="W&B project name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_adamw_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay = []
    no_decay = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    name = config.name.lower()
    betas = tuple(config.betas)

    if name == "adamw":
        return torch.optim.AdamW(
            build_adamw_param_groups(model, config.weight_decay),
            lr=config.lr,
            betas=betas,
            eps=config.eps,
        )

    if name == "adam":
        return torch.optim.Adam(
            build_adamw_param_groups(model, config.weight_decay),
            lr=config.lr,
            betas=betas,
            eps=config.eps,
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {config.name}")


def _cosine_lr_lambda(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)

    if total_steps <= warmup_steps:
        return 1.0

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    total_steps: int,
) -> LambdaLR:
    name = config.name.lower()

    if name in {"none", "constant"}:
        return LambdaLR(optimizer, lambda _: 1.0)

    if name == "cosine":
        return LambdaLR(
            optimizer,
            lambda step: _cosine_lr_lambda(
                step,
                warmup_steps=config.warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=config.min_lr_ratio,
            ),
        )

    raise ValueError(f"Unsupported scheduler: {config.name}")


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()

    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


def make_dataloader(
    dataset: PackedPretrainDataset,
    config: TrainConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn,
        generator=generator if shuffle else None,
    )


def infer_total_steps(
    train_dataset: PackedPretrainDataset,
    train_loader: DataLoader,
    config: TrainConfig,
    *,
    start_step: int = 0,
    start_tokens: int = 0,
    target_tokens: int | None = None,
) -> int:
    if target_tokens is not None:
        if target_tokens <= start_tokens:
            raise ValueError(
                f"target_tokens={target_tokens} must be greater than "
                f"start_tokens={start_tokens}"
            )
        batch_tokens = (
            config.training.batch_size
            * config.model.max_seq_len
            * config.training.gradient_accumulation_steps
        )
        additional_steps = math.ceil((target_tokens - start_tokens) / batch_tokens)
        return start_step + additional_steps

    if config.training.max_steps is not None:
        return config.training.max_steps

    steps_per_pass = len(train_loader) // config.training.gradient_accumulation_steps
    if steps_per_pass > 0:
        return steps_per_pass

    return max(1, train_dataset.meta["training_tokens"] // config.model.max_seq_len)


def batch_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    mixed_precision: str,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    with autocast_context(device, mixed_precision):
        logits = model(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    max_steps: int,
) -> dict[str, float]:
    model.eval()
    losses = []
    tokens = 0

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break

        loss = batch_loss(model, batch, device, mixed_precision)
        losses.append(float(loss.detach().cpu()))
        tokens += batch["input_ids"].numel()

    model.train()
    if not losses:
        return {"validation_loss": float("nan"), "validation_tokens": 0}

    mean_loss = sum(losses) / len(losses)
    return {
        "validation_loss": mean_loss,
        "validation_ppl": math.exp(min(20.0, mean_loss)),
        "validation_tokens": tokens,
    }


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    max_seq_len: int,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    model.eval()
    samples = []

    for prompt in DEFAULT_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids

        for _ in range(max_new_tokens):
            context = generated_ids[:, -max_seq_len:]
            logits = model(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        samples.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                ),
            }
        )

    model.train()
    return samples


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: TrainConfig,
    global_step: int,
    tokens_seen: int,
    train_dataset: PackedPretrainDataset,
    validation_dataset: PackedPretrainDataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
        "train_dataset_meta": train_dataset.meta,
        "validation_dataset_meta": validation_dataset.meta,
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    try:
        torch.random.set_rng_state(checkpoint["torch_rng_state"].cpu())
    except (TypeError, RuntimeError, AttributeError) as exc:
        print(f"Warning: failed to restore CPU RNG state: {exc}")
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        try:
            cuda_states = [
                state.cpu().to(torch.uint8)
                if isinstance(state, torch.Tensor)
                else torch.as_tensor(state, dtype=torch.uint8)
                for state in checkpoint["cuda_rng_state_all"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
        except (TypeError, RuntimeError, AttributeError) as exc:
            print(f"Warning: failed to restore CUDA RNG state: {exc}")
    return int(checkpoint["global_step"]), int(checkpoint["tokens_seen"])


def main() -> None:
    args = parse_args()
    config = TrainConfig.load_from_yaml(args.config_path)
    set_seed(config.training.seed)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(**asdict(config.tokenizer))
    train_dataset = PackedPretrainDataset(
        args.dataset_path,
        tokenizer,
        config.model.max_seq_len,
        split="train",
        rebuild_cache=config.data.rebuild_cache,
        parquet_batch_size=config.data.parquet_batch_size,
        tokenize_batch_size=config.data.tokenize_batch_size,
        token_budget=config.data.token_budget,
    )
    validation_dataset = PackedPretrainDataset(
        args.dataset_path,
        tokenizer,
        config.model.max_seq_len,
        split="validation",
        rebuild_cache=config.data.rebuild_cache,
        parquet_batch_size=config.data.parquet_batch_size,
        tokenize_batch_size=config.data.tokenize_batch_size,
        token_budget=config.data.validation_token_budget,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelFactory.build_model_from_config(config.model).to(device)
    if config.training.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loader = make_dataloader(train_dataset, config, shuffle=True)
    validation_loader = make_dataloader(validation_dataset, config, shuffle=False)

    global_step = 0
    tokens_seen = 0
    initial_total_steps = infer_total_steps(train_dataset, train_loader, config)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, initial_total_steps)
    resume_from = (
        Path(config.training.resume_from)
        if config.training.resume_from
        else None
    )
    if resume_from is not None:
        global_step, tokens_seen = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    total_steps = infer_total_steps(
        train_dataset,
        train_loader,
        config,
        start_step=global_step,
        start_tokens=tokens_seen,
        target_tokens=config.training.target_tokens,
    )
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps)
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    run = wandb.init(
        project=args.wandb,
        name=args.run_name or config.training.run_name,
        mode=args.wandb_mode,
        config={
            **asdict(config),
            "dataset_path": str(args.dataset_path),
            "train_dataset_meta": train_dataset.meta,
            "validation_dataset_meta": validation_dataset.meta,
            "total_steps": total_steps,
            "target_tokens": config.training.target_tokens,
            "resume_from": str(resume_from) if resume_from else None,
        },
    )
    run_name = args.run_name or config.training.run_name or run.name or "run"
    run_weights_dir = args.weights_dir / run_name
    run_weights_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)
    model.train()

    train_iterator = iter(train_loader)
    progress = tqdm(
        total=total_steps,
        initial=global_step,
        desc="pretrain",
        dynamic_ncols=True,
    )

    while global_step < total_steps:
        accumulated_loss = 0.0
        accumulated_tokens = 0

        for _ in range(config.training.gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            loss = batch_loss(
                model,
                batch,
                device,
                config.training.mixed_precision,
            )
            (loss / config.training.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach().cpu())
            accumulated_tokens += batch["input_ids"].numel()

        if config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.max_grad_norm,
            )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        tokens_seen += accumulated_tokens
        progress.update(1)

        train_loss = accumulated_loss / config.training.gradient_accumulation_steps
        progress.set_postfix(loss=f"{train_loss:.4f}", lr=scheduler.get_last_lr()[0])

        if global_step % config.training.log_interval_steps == 0:
            wandb.log(
                {
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )

        if global_step % config.training.eval_interval_steps == 0:
            metrics = evaluate(
                model,
                validation_loader,
                device,
                config.training.mixed_precision,
                config.training.eval_steps,
            )
            metrics.update({"step": global_step, "tokens_seen": tokens_seen})
            wandb.log(metrics, step=global_step)

        if global_step % config.training.sample_interval_steps == 0:
            samples = generate_samples(
                model,
                tokenizer,
                device,
                config.model.max_seq_len,
                config.training.max_new_tokens,
            )
            table = wandb.Table(columns=["prompt", "completion"])
            for sample in samples:
                table.add_data(sample["prompt"], sample["completion"])
            wandb.log({"generated_samples": table}, step=global_step)

        if global_step % config.training.checkpoint_interval_steps == 0:
            checkpoint_path = run_weights_dir / f"checkpoint-step-{global_step}.pt"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                global_step=global_step,
                tokens_seen=tokens_seen,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
            )
            latest_path = run_weights_dir / "latest.pt"
            shutil.copy2(checkpoint_path, latest_path)

    progress.close()

    final_metrics = evaluate(
        model,
        validation_loader,
        device,
        config.training.mixed_precision,
        config.training.eval_steps,
    )
    final_metrics.update({"step": global_step, "tokens_seen": tokens_seen})
    wandb.log(final_metrics, step=global_step)

    final_checkpoint = run_weights_dir / f"checkpoint-step-{global_step}.pt"
    save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        global_step=global_step,
        tokens_seen=tokens_seen,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    shutil.copy2(final_checkpoint, run_weights_dir / "latest.pt")
    run.finish()


if __name__ == "__main__":
    main()
