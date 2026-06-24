import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.positional_encoding import SinusoidalPositionalEncoding
from src.model.decoder import DecoderBlock, DecoderBlockV1_5, DecoderBlockV2
from src.model.normalize import RMSNorm
from src import ModelFactory


@ModelFactory.register_model("MyGPT")
class MyGPT(nn.Module):
  def __init__(self, d_model,
               vocab_size,
               n_decoder_blocks,
               n_attention_heads=8,
               dropout=0.1,
               max_seq_len=512):
    super(MyGPT, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
    decoders = [
        DecoderBlock(
            d_model,
            n_attention_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        for _ in range(n_decoder_blocks)
    ]
    self.decoders = nn.Sequential(*decoders)
    self.output_layer = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, vocab_size)
    )
  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = self.decoders(x)
    x = self.output_layer(x)
    return x


@ModelFactory.register_model("MyGPT2")
class MyGPT1_5(nn.Module):
  def __init__(self, d_model,
               vocab_size,
               n_decoder_blocks,
               n_attention_heads=8,
               dropout=0.1,
               max_seq_len=512,
               tie_embeddings=True):
    super(MyGPT1_5, self).__init__()
    self.n_decoder_blocks = n_decoder_blocks
    self.tie_embeddings = tie_embeddings
    self.embedding = nn.Embedding(vocab_size, d_model)
    decoders = [
        DecoderBlockV1_5(
            d_model,
            n_attention_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        for _ in range(n_decoder_blocks)
    ]
    self.decoders = nn.Sequential(*decoders)
    self.normalize = RMSNorm(d_model)
    if not self.tie_embeddings:
      self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    self.apply(self._init_weights)
    self._init_residual_projections()

  @staticmethod
  def _init_weights(module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def _init_residual_projections(self):
    residual_std = 0.02 / math.sqrt(2 * self.n_decoder_blocks)
    for decoder in self.decoders:
      nn.init.normal_(decoder.attention.linear.weight, mean=0.0, std=residual_std)
      nn.init.normal_(decoder.ffn.w_o.weight, mean=0.0, std=residual_std)

  def forward(self, x):
    x = self.embedding(x)
    x = self.decoders(x)
    x = self.normalize(x)
    if not self.tie_embeddings:
      return self.lm_head(x)
    return F.linear(x, self.embedding.weight)


@ModelFactory.register_model("MyGPT2")
class MyGPT2(nn.Module):
  def __init__(self, d_model,
               vocab_size,
               n_decoder_blocks,
               n_attention_heads=8,
               dropout=0.1,
               max_seq_len=512,
               tie_embeddings=True):
    super(MyGPT2, self).__init__()
    self.n_decoder_blocks = n_decoder_blocks
    self.tie_embeddings = tie_embeddings
    self.embedding = nn.Embedding(vocab_size, d_model)
    decoders = [
        DecoderBlockV2(
            d_model,
            n_attention_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        for _ in range(n_decoder_blocks)
    ]
    self.decoders = nn.Sequential(*decoders)
    self.normalize = RMSNorm(d_model)
    if not self.tie_embeddings:
      self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    self.apply(self._init_weights)
    self._init_residual_projections()

  @staticmethod
  def _init_weights(module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def _init_residual_projections(self):
    residual_std = 0.02 / math.sqrt(2 * self.n_decoder_blocks)
    for decoder in self.decoders:
      nn.init.normal_(decoder.attention.linear.weight, mean=0.0, std=residual_std)
      nn.init.normal_(decoder.ffn.w_o.weight, mean=0.0, std=residual_std)

  def forward(self, x):
    x = self.embedding(x)
    x = self.decoders(x)
    x = self.normalize(x)
    if not self.tie_embeddings:
      return self.lm_head(x)
    return F.linear(x, self.embedding.weight)
