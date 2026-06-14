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
  --dataset_path beomi/KoAlpaca-v1.1a \
  --weights_dir ./checkpoints \
  --config_path ./config.yaml
```

Weights, datasets, local environments, Weights & Biases logs, and local editor
settings are excluded from Git.

## External resources

The default configuration downloads `skt/kogpt2-base-v2`, and the example
command downloads `beomi/KoAlpaca-v1.1a`. Review and follow the terms of those
resources before redistributing models, datasets, or generated weights.

## License

No open-source license has been granted for this repository yet.
