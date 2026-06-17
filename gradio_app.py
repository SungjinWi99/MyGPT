import os
from pathlib import Path

import gradio as gr
import torch

from src.inference import generate_text, load_model_from_checkpoint


DEFAULT_CHECKPOINT = os.environ.get(
    "MYGPT_CHECKPOINT",
    "/content/drive/MyDrive/KTB/MyGPT/checkpoints/8layer-2_4b/latest.pt",
)
DEFAULT_CONFIG = os.environ.get("MYGPT_CONFIG", "./config.8layer_2_4b.yaml")

MODEL_CACHE = {}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_cached_model(
    checkpoint_path: str,
    config_path: str,
    device: str,
    prefer_checkpoint_config: bool,
):
    key = (checkpoint_path, config_path, device, prefer_checkpoint_config)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = load_model_from_checkpoint(
            checkpoint_path,
            config_path,
            device=device,
            prefer_checkpoint_config=prefer_checkpoint_config,
        )
    return MODEL_CACHE[key]


def generate(
    checkpoint_path: str,
    config_path: str,
    prefer_checkpoint_config: bool,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[str, str]:
    resolved_device = default_device() if device == "auto" else device

    if not Path(checkpoint_path).exists():
        return "", f"Checkpoint file does not exist: {checkpoint_path}"

    try:
        model, tokenizer, config, checkpoint = load_cached_model(
            checkpoint_path,
            config_path,
            resolved_device,
            prefer_checkpoint_config,
        )
        output = generate_text(
            model,
            tokenizer,
            prompt,
            max_seq_len=config.model.max_seq_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            device=resolved_device,
        )
        metadata = (
            f"device={resolved_device}\n"
            f"global_step={checkpoint.get('global_step')}\n"
            f"tokens_seen={checkpoint.get('tokens_seen')}\n"
            f"model={config.model.model_name}, "
            f"d_model={config.model.d_model}, "
            f"layers={config.model.n_decoder_blocks}\n"
            f"temperature={temperature}, top_k={top_k}, top_p={top_p}, "
            f"repetition_penalty={repetition_penalty}, "
            f"no_repeat_ngram_size={int(no_repeat_ngram_size)}"
        )
        return output, metadata
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


with gr.Blocks(title="MyGPT Demo") as demo:
    gr.Markdown("# MyGPT Demo")
    gr.Markdown("Load a trained checkpoint and test Korean text generation.")

    with gr.Accordion("Checkpoint", open=True):
        checkpoint_path = gr.Textbox(
            label="Checkpoint path",
            value=DEFAULT_CHECKPOINT,
        )
        config_path = gr.Textbox(label="Config path", value=DEFAULT_CONFIG)
        prefer_checkpoint_config = gr.Checkbox(
            label="Use config embedded in checkpoint",
            value=True,
        )
        device = gr.Radio(
            label="Device",
            choices=["auto", "cpu", "cuda"],
            value="auto",
        )

    prompt = gr.Textbox(
        label="Prompt",
        value="한국어 인공지능 모델을 직접 학습하는 이유는",
        lines=4,
    )

    with gr.Row():
        max_new_tokens = gr.Slider(8, 512, value=128, step=8, label="Max new tokens")
        temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
    with gr.Row():
        top_k = gr.Slider(0, 200, value=50, step=5, label="Top-k")
        top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="Top-p")
    with gr.Row():
        repetition_penalty = gr.Slider(
            1.0,
            2.0,
            value=1.15,
            step=0.05,
            label="Repetition penalty",
        )
        no_repeat_ngram_size = gr.Slider(
            0,
            8,
            value=3,
            step=1,
            label="No repeat n-gram size",
        )

    generate_button = gr.Button("Generate", variant="primary")
    output = gr.Textbox(label="Generated text", lines=10)
    metadata = gr.Textbox(label="Metadata / errors", lines=5)

    generate_button.click(
        generate,
        inputs=[
            checkpoint_path,
            config_path,
            prefer_checkpoint_config,
            device,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
        ],
        outputs=[output, metadata],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
