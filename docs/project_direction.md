# MyGPT Project Direction

## Goal

The project goal is to improve the quality of a personal Korean chatbot through
repeatable training experiments and evidence-based iteration.

The project is also explicitly educational. Training must use the model
implementation in this repository and start from randomly initialized model
weights. Existing pretrained model weights are out of scope. Reusing an external
tokenizer or tokenizer vocabulary does not imply loading that model's weights.

The first pretraining and SFT dataset versions use the existing
`skt/kogpt2-base-v2` tokenizer. Its exact Hugging Face revision and effective
special-token configuration must be pinned in the manifest. Training a custom
tokenizer is deferred to a later controlled experiment.

## Working Agreement

- Codex develops and maintains the code in the local repository.
- The user clones or pulls the repository in Google Colab and starts experiments
  manually.
- Colab is the execution environment; Google Drive stores reusable datasets and
  training checkpoints.
- The user tells Codex when an experiment is complete.
- Codex reads the corresponding public W&B run and summarizes training metrics,
  generated samples, configuration, runtime, and relevant system metrics.
- The user and Codex use those results to discuss the next performance
  improvement.

W&B project:

- <https://wandb.ai/dnltjdwls1/MyGPT?nw=nwuserdnltjdwls1>

## Initial Improvement Direction

The first priority is improving the quantity and quality of Korean training
data. Training is split into two distinct stages:

1. Pretrain the custom decoder-only model on general Korean corpora with
   next-token prediction.
2. Supervised fine-tune the resulting checkpoint on Korean single-turn
   instruction-response and question-answer data.

The intended dataset workflow is:

1. Find suitable Korean datasets, primarily on Hugging Face.
2. Review their task fit, quality, size, and license before inclusion.
3. Convert each source dataset with a source-specific adapter.
4. Normalize all accepted records into one common schema.
5. Apply validation, filtering, deduplication, and deterministic train/validation
   splits.
6. Save the processed dataset to a separate Google Drive directory.
7. Reuse the processed dataset across experiments without repeating the full
   preprocessing step.

The first dataset version targets general-purpose Korean single-turn
instruction following and question answering. Multi-turn conversations are
excluded from this phase.

Only originally authored Korean data is included in the first dataset version.
Translated and synthetic datasets are deferred and should be tracked separately
rather than mixed into the initial training corpus.

The first version does not apply topic-based or safety-category filtering.
Technical validation such as removing empty, malformed, or exact duplicate
records still applies because it directly affects training quality.

The initial deduplication policy removes exact duplicates only. Near-duplicate
methods, source-specific quality filters, document-length thresholds, boilerplate
removal, and detailed mixing rules are intentionally deferred until candidate
datasets have been selected, profiled, and sampled manually. Those policies
must be based on observed source characteristics rather than decided in the
abstract.

Research-only and personal-use datasets may be included. Every source must
retain its license and usage restrictions in the generated manifest. A combined
dataset or trained checkpoint that includes restricted sources must not be
treated as redistributable or commercially usable without a separate license
review.

## Dataset Pipeline Direction

The current `KoAlpacaDataset` implementation assumes one dataset's field layout
and reads values by dictionary order. That contract is too fragile for combining
multiple datasets.

The combined dataset pipeline should instead produce explicit, versioned fields.
The exact schema will be finalized through the requirements interview. A likely
instruction-tuning schema is:

```text
id
source
instruction
input
output
license
metadata
```

The pipeline should keep these responsibilities separate:

- source loading
- source-specific field conversion
- text normalization and quality filtering
- duplicate detection
- dataset mixing or sampling
- split generation
- persistence and manifest generation

General corpora and supervised instruction data must remain separate dataset
products because they use different objectives, schemas, and training stages.

Each generated dataset version should include a manifest containing source
revisions, record counts, filtering statistics, schema version, build
configuration, and a reproducible fingerprint.

## Experiment Loop

1. Codex prepares code and the exact Colab commands.
2. The user pulls the repository and runs the experiment in Colab.
3. The user reports completion and provides the W&B run link when needed.
4. Codex collects and summarizes the W&B evidence.
5. The user and Codex decide the next controlled change.
6. Codex implements that change locally.

Experiments should change a limited number of variables at a time so that gains
can be attributed to a specific dataset, filter, mixture, or training change.

## Compute Constraint

Training uses one NVIDIA A100 80 GB GPU in Colab. The current architecture is
roughly 65 million parameters with a 51,200-token vocabulary, so GPU memory is
not the primary constraint; total token throughput and Colab runtime are.

Each main training run should target an 8-12 hour maximum wall-clock duration.
The token budget for a run is calculated from measured pilot throughput and must
fit within that duration with time reserved for validation and checkpoint
writes. Longer training is continued as a new resumable run rather than relying
on one uninterrupted Colab session.

Pretraining progress should be measured in tokens rather than epochs. The
initial planning milestones are:

1. Smoke test: 5-10 million tokens to validate data, loss, checkpoints, and
   resume behavior.
2. Pilot: 100 million tokens to validate learning curves and generation.
3. First main run: 300-500 million tokens if the pilot remains stable.
4. Stretch run: up to 1 billion tokens only if validation loss is still
   improving and measured throughput makes the runtime acceptable.

The corpus builder may prepare more data than a run consumes. Each experiment
must record the exact token budget and sampled data fingerprint.

Google Drive has approximately 4 TB available, so dataset storage capacity is
not a practical constraint. Each reusable dataset version should use:

- sharded Parquet files for cleaned, normalized source text
- a separate tokenized and fixed-length packed cache for training throughput
- `manifest.json` for source revisions, licenses, schema, tokenizer identity,
  filtering and deduplication statistics, split information, token counts, and
  fingerprints

The Google Drive dataset root is:

```text
/content/drive/MyDrive/KTB/MyGPT/datasets/
```

Initial version layout:

```text
datasets/
  pretrain/
    v1/
      parquet/
      tokenized/
      manifest.json
  sft/
    v1/
      parquet/
      tokenized/
      manifest.json
```

The cleaned Parquet dataset remains the canonical reusable source. Token caches
are derived artifacts and must be rebuilt when the tokenizer, sequence length,
packing policy, or source fingerprint changes.

Dataset versions are immutable after a successful build. Each manifest pins the
exact Hugging Face dataset revision or source snapshot. Adding a source or
changing a source revision, schema, filter, deduplication rule, split policy,
tokenizer, sequence length, or packing policy creates a new version instead of
overwriting an existing one. Failed or incomplete builds are not promoted as
released versions.

Pretraining data uses a deterministic document-level split:

- 99.5% training
- 0.5% validation

The split should approximately preserve source-category proportions. A document
must never cross splits, and duplicate or near-duplicate leakage between train
and validation must be checked. Once generated, the validation set remains fixed
across comparable model experiments. Generation quality is evaluated separately
with a fixed Korean prompt suite.

The preferred pretraining corpus categories are Korean encyclopedia/wiki text,
news, and public or institutional documents. General web and blog text is not a
default source. It may be considered later only if the preferred sources cannot
supply enough usable tokens after technical filtering and deduplication.

The current v1 profiling build uses the Korean Wikimedia dump and
`HAERAE-HUB/KOREAN-WEBTEXT`. Open Korean Historical Corpus was removed after
the smoke candidate produced no accepted records under the intended filter and
was found to be dominated by Hanja and old-Hangul material. NIKL remains
deferred.

The preferred-source corpus is considered sufficient when it contains at least
500 million usable tokens after filtering and deduplication. If it falls below
that threshold, search for additional public or institutional sources before
considering general web or blog data.

Final source mixing ratios are chosen after candidate discovery and
normalization, using post-filter token counts and observed quality. As an
initial review rule, no single source should exceed 50% of training tokens
unless there is a documented reason and a comparison experiment supporting the
choice.

Before a main pretraining run, the training path should support:

- packed fixed-length token blocks without padding waste
- BF16 automatic mixed precision
- PyTorch scaled dot-product causal attention
- gradient accumulation and gradient clipping
- warmup plus learning-rate decay
- held-out validation loss
- step-based checkpointing and exact resume
- W&B logging for tokens processed, tokens per second, learning rate, train
  loss, validation loss, GPU memory, and elapsed time

## Automated Evaluation

Checkpoint evaluation is automated in the training code. The initial fixed
generation suite contains 30 Korean prompts, with five prompts in each category:

- factual or common-knowledge question answering
- explanation
- summarization
- reasoning
- creative generation
- instruction following

Prompt text, decoding parameters, and random seeds are versioned and fixed for
comparable experiments. Every 5,000 optimizer steps, the trainer runs validation
loss and a short generation check. At the end of each run, it evaluates the full
30-prompt suite and logs results to a W&B Table.

The 5,000-step interval is an initial default rather than a permanent physical
unit. W&B must also record cumulative training tokens and tokens per optimizer
step so that evaluation cadence remains interpretable when batch size, sequence
length, or gradient accumulation changes.

Automated generation diagnostics should include response length, empty or
invalid response rate, repetition, Korean-character ratio, prompt-copy ratio,
and termination behavior. Codex reviews the W&B generations qualitatively for
relevance, fluency, instruction following, and coherence. The user only needs
to provide preference judgments when a comparison is ambiguous or subjective.

There is no fixed universal loss threshold that automatically declares an
experiment successful. After every completed experiment, the user and Codex
review W&B evidence together, including validation-loss trends, fixed-prompt
generations, automated diagnostics, runtime, and comparisons with prior
checkpoints. They then decide whether to continue training, change data, change
optimization, or revise the model. Automated metrics support this decision but
do not replace it.

## Decisions Pending Interview

- Separate normalized schemas for pretraining text and single-turn SFT data
- Source-specific quality filters and any near-duplicate policy, after dataset
  selection and analysis
- Parquet and token-cache shard sizes
- Exact fixed generation prompt contents and comparison baseline
