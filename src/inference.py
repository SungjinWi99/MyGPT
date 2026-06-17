from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from dacite import from_dict
from transformers import PreTrainedTokenizerFast

import src.model  # noqa: F401 - registers model classes in ModelFactory.
from src.config import TrainConfig, ModelFactory


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def load_config_for_checkpoint(
    config_path: str | Path,
    checkpoint: dict[str, Any],
    *,
    prefer_checkpoint_config: bool = True,
) -> TrainConfig:
    if prefer_checkpoint_config and isinstance(checkpoint.get("config"), dict):
        return from_dict(data_class=TrainConfig, data=checkpoint["config"])
    return TrainConfig.load_from_yaml(str(config_path))


def load_tokenizer(config: TrainConfig) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(**asdict(config.tokenizer))


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    prefer_checkpoint_config: bool = True,
) -> tuple[torch.nn.Module, PreTrainedTokenizerFast, TrainConfig, dict[str, Any]]:
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = load_config_for_checkpoint(
        config_path,
        checkpoint,
        prefer_checkpoint_config=prefer_checkpoint_config,
    )

    tokenizer = load_tokenizer(config)
    model = ModelFactory.build_model_from_config(config.model).to(device)
    state_dict = _strip_compile_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer, config, checkpoint


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        remove_indices = sorted_indices[sorted_mask]
        logits[..., remove_indices] = float("-inf")

    return logits


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    *,
    max_seq_len: int,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device | str = "cpu",
) -> str:
    device = torch.device(device)
    generated_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        context = generated_ids[:, -max_seq_len:]
        logits = model(context)[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
