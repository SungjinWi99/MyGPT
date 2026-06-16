from dataclasses import dataclass, asdict, field
import yaml
from dacite import from_dict  # 딕셔너리 -> 데이터클래스 변환기


@dataclass
class ModelConfig:
    model_name: str
    d_model: int
    vocab_size: int
    n_decoder_blocks: int
    n_attention_heads: int
    max_seq_len: int = 1024

@dataclass
class TokenizerConfig:
    pretrained_model_name_or_path: str = "skt/kogpt2-base-v2"
    bos_token: str = '</s>'
    eos_token: str = '</s>'
    unk_token: str = '<unk>'
    pad_token: str = '<pad>'
    mask_token: str = '<mask>'


@dataclass
class DataConfig:
    token_budget: int | None = None
    validation_token_budget: int | None = None
    rebuild_cache: bool = False
    parquet_batch_size: int = 1024
    tokenize_batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 0.0004
    weight_decay: float = 0.1
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int | None = None
    max_grad_norm: float = 1.0
    log_interval_steps: int = 10
    eval_interval_steps: int = 500
    eval_steps: int = 100
    checkpoint_interval_steps: int = 1000
    sample_interval_steps: int = 500
    max_new_tokens: int = 100
    seed: int = 42
    mixed_precision: str = "bf16"
    compile_model: bool = False


@dataclass
class TrainConfig:
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return from_dict(data_class=cls, data=config_dict)


class ModelFactory:
    _registry = {}

    @classmethod
    def register_model(cls, model_name):
        def wrapper(model_cls):
            cls._registry[model_name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def build_model_from_config(cls, cfg: ModelConfig):
        cfg_dict = asdict(cfg)
        model_name = cfg_dict.pop('model_name')
        if model_name not in cls._registry:
            raise ValueError(f"Unexpected Model Name: {model_name}")
        model_cls = cls._registry[model_name]
        return model_cls(**cfg_dict)
