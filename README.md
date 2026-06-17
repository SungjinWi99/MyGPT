# MyGPT

A small decoder-only Transformer language model implemented with PyTorch and
trained on Korean instruction data.

See [docs/project_direction.md](docs/project_direction.md) for the project goal,
Colab/W&B workflow, and the initial dataset improvement direction.
See [docs/dataset_candidates.md](docs/dataset_candidates.md) for the first
Korean pretraining and SFT dataset review.
The [Korean HTML version](docs/dataset_candidates_ko.html) is available for
browser viewing.
See [docs/data_preparation.md](docs/data_preparation.md) for the accepted
pretraining sources, Colab commands, and Google Drive layout.
See [docs/model_architecture_improvements_ko.html](docs/model_architecture_improvements_ko.html)
for the Korean implementation note covering the planned model architecture
modernization.
See [docs/pretrain_dataset_code_ko.html](docs/pretrain_dataset_code_ko.html)
for the Korean explanation of the packed pretraining dataset and training input
code.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Review `config.yaml`, then run:

```bash
python -m src.train \
  --dataset-path /content/drive/MyDrive/KTB/MyGPT/datasets/pretrain/v1 \
  --weights-dir /content/drive/MyDrive/KTB/MyGPT/checkpoints \
  --config-path ./config.yaml \
  --wandb MyGPT
```

`scripts/prepare_pretrain_colab.sh` writes `pretrain/v1` by default. If you built
another version, pass that version explicitly:

```bash
DATASET_VERSION=v2 ./train.sh
```

To continue the latest 400M-token run up to 800M cumulative tokens, use the
dedicated config:

```bash
DATASET_VERSION=v2 CONFIG_PATH=./config.continue_800m.yaml ./train.sh
```

Continuation settings such as `resume_from`, `target_tokens`, and `run_name`
live in the config file under `training`.

## Demo

For Colab, prefer the Gradio demo because it provides a public share URL without
an extra tunnel:

```bash
python gradio_app.py
```

It prints a `https://*.gradio.live` URL. Keep the cell running while using the
demo. By default it loads the latest 8-layer run:

```text
/content/drive/MyDrive/KTB/MyGPT/checkpoints/8layer-2_4b/latest.pt
```

To test another checkpoint, set the path before launching:

```bash
MYGPT_CHECKPOINT=/content/drive/MyDrive/KTB/MyGPT/checkpoints/your-run/latest.pt \
MYGPT_CONFIG=./config.yaml \
python gradio_app.py
```

You can also run the Streamlit demo locally:

```bash
streamlit run streamlit_app.py
```

In Colab, the checkpoint path usually looks like:

```text
/content/drive/MyDrive/KTB/MyGPT/checkpoints/8layer-2_4b/latest.pt
```

Use that `latest.pt` path in the Streamlit sidebar.

Weights, datasets, local environments, Weights & Biases logs, and local editor
settings are excluded from Git.

## External resources

The default configuration downloads the `skt/kogpt2-base-v2` tokenizer. The
training dataset is the local cleaned Parquet corpus produced by
`src.dataset_pipeline.prepare_pretrain`. Review and follow the terms of all
source datasets before redistributing models, datasets, or generated weights.

## License

No open-source license has been granted for this repository yet.
