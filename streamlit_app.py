from pathlib import Path

import streamlit as st
import torch

from src.inference import generate_text, load_model_from_checkpoint


st.set_page_config(page_title="MyGPT Demo", page_icon="M", layout="wide")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading checkpoint...")
def cached_load_model(
    checkpoint_path: str,
    config_path: str,
    device: str,
    prefer_checkpoint_config: bool,
):
    return load_model_from_checkpoint(
        checkpoint_path,
        config_path,
        device=device,
        prefer_checkpoint_config=prefer_checkpoint_config,
    )


st.title("MyGPT Streamlit Demo")
st.caption("Load a MyGPT checkpoint and test Korean text generation.")

with st.sidebar:
    st.header("Checkpoint")
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value="/content/drive/MyDrive/KTB/MyGPT/checkpoints/latest.pt",
        help="Use the latest.pt or checkpoint-step-*.pt file created by src.train.",
    )
    config_path = st.text_input("Config path", value="./config.yaml")
    prefer_checkpoint_config = st.checkbox(
        "Use config embedded in checkpoint",
        value=True,
        help="Recommended. Falls back to config path when checkpoint has no config.",
    )

    st.header("Generation")
    device = st.selectbox(
        "Device",
        options=["auto", "cpu", "cuda"],
        index=0,
    )
    resolved_device = default_device() if device == "auto" else device
    max_new_tokens = st.slider("Max new tokens", 8, 512, 128, step=8)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.8, step=0.05)
    top_k = st.slider("Top-k", 0, 200, 50, step=5)
    top_p = st.slider("Top-p", 0.05, 1.0, 0.95, step=0.05)

prompt = st.text_area(
    "Prompt",
    value="한국어 인공지능 모델을 직접 학습하는 이유는",
    height=140,
)

checkpoint_exists = Path(checkpoint_path).exists()
config_exists = Path(config_path).exists()

col1, col2 = st.columns(2)
col1.metric("Checkpoint", "found" if checkpoint_exists else "missing")
col2.metric("Device", resolved_device)

if not checkpoint_exists:
    st.warning("Checkpoint file does not exist. Check the path in the sidebar.")
if not config_exists and not prefer_checkpoint_config:
    st.warning("Config file does not exist and checkpoint config is disabled.")

if st.button("Generate", type="primary", disabled=not checkpoint_exists):
    try:
        model, tokenizer, config, checkpoint = cached_load_model(
            checkpoint_path,
            config_path,
            resolved_device,
            prefer_checkpoint_config,
        )
        with st.spinner("Generating..."):
            output = generate_text(
                model,
                tokenizer,
                prompt,
                max_seq_len=config.model.max_seq_len,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=resolved_device,
            )

        st.subheader("Output")
        st.text_area("Generated text", value=output, height=260)

        with st.expander("Checkpoint metadata"):
            st.json(
                {
                    "global_step": checkpoint.get("global_step"),
                    "tokens_seen": checkpoint.get("tokens_seen"),
                    "model": checkpoint.get("config", {}).get("model"),
                    "tokenizer": checkpoint.get("config", {}).get("tokenizer"),
                }
            )
    except Exception as exc:
        st.exception(exc)
