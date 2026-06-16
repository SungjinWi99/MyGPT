import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.positional_encoding import SinusoidalPositionalEncoding
from src.model.decoder import DecoderBlock, DecoderBlockV2
from src.model.normalize import RMSNorm
from src import ModelFactory


@ModelFactory.register_model("MyGPT")
class MyGPT(nn.Module):
  def __init__(self, d_model, vocab_size, n_decoder_blocks, n_attention_heads=8, max_seq_len=512):
    super(MyGPT, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
    decoders = [DecoderBlock(d_model, n_attention_heads, max_seq_len) for _ in range(n_decoder_blocks)]
    self.decoders = nn.Sequential(*decoders)
    self.output_layer =nn.Sequential(
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
class MyGPT2(nn.Module):
  def __init__(self, d_model,
               vocab_size,
               n_decoder_blocks,
               n_attention_heads=8,
               max_seq_len=512):
    super(MyGPT2, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    decoders = [DecoderBlockV2(d_model, n_attention_heads, max_seq_len) for _ in range(n_decoder_blocks)]
    self.decoders = nn.Sequential(*decoders)
    self.normalize = RMSNorm(d_model)

  def forward(self, x):
    x = self.embedding(x)
    x = self.decoders(x)
    x = self.normalize(x)
    return F.linear(x, self.embedding.weight)
