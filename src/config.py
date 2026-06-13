from dataclasses import dataclass
import yaml
from dacite import from_dict  # 딕셔너리 -> 데이터클래스 변환기


@dataclass
class ModelConfig:
    d_model: int
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
class TrainConfig:
    model: ModelConfig
    tokenizer: TokenizerConfig
    lr: float
    epochs: int
    batch_size: int

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return from_dict(data_class=cls, data=config_dict)