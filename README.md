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

Weights, datasets, local environments, Weights & Biases logs, and local editor
settings are excluded from Git.

## External resources

The default configuration downloads the `skt/kogpt2-base-v2` tokenizer. The
training dataset is the local cleaned Parquet corpus produced by
`src.dataset_pipeline.prepare_pretrain`. Review and follow the terms of all
source datasets before redistributing models, datasets, or generated weights.

## License

No open-source license has been granted for this repository yet.
