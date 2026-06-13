import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads, max_seq_len=512):
    super(MultiHeadAttention, self).__init__()
    self.n_heads = n_heads
    self.q = nn.Linear(d_model, d_model)
    self.k = nn.Linear(d_model, d_model)
    self.v = nn.Linear(d_model, d_model)
    self.linear = nn.Linear(d_model, d_model)
    self.max_seq_len = max_seq_len
    self.register_buffer('attention_mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))

  def forward(self, x):
    B, T, C = x.size()

    q = self.q(x)
    q = q.view(B, T, self.n_heads, C // self.n_heads)
    q = q.transpose(1,2)

    k = self.k(x)
    k = k.view(B, T, self.n_heads, C // self.n_heads)
    k = k.transpose(1,2)

    v = self.v(x)
    v = v.view(B, T, self.n_heads, C // self.n_heads)
    v = v.transpose(1,2)

    attention_weights = (q @ k.transpose(2,3)) / math.sqrt(q.size(-1))
    attention_weights = attention_weights.masked_fill(self.attention_mask[:T,:T]==0, value=float('-inf'))
    attention_weights = F.softmax(attention_weights, dim=-1)
    return self.linear((attention_weights @ v).transpose(1,2).contiguous().view(B, T, C))

class DecoderBlock(nn.Module):
  def __init__(self, d_model, n_attention_heads, max_seq_len=512):
    super(DecoderBlock, self).__init__()
    self.normalize1 = nn.LayerNorm(d_model)
    self.attention = MultiHeadAttention(d_model, n_attention_heads, max_seq_len)
    self.normalize2 = nn.LayerNorm(d_model)
    self.ffn = nn.Sequential(
        nn.Linear(d_model, d_model*4),
        nn.GELU(),
        nn.Linear(d_model*4, d_model)
    )
  def forward(self, x):
    x = x + self.attention(self.normalize1(x))
    return x + self.ffn(self.normalize2(x))

class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_len):
    super(SinusoidalPositionalEncoding, self).__init__()
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, 1).unsqueeze(1)
    d = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:,0::2] = torch.sin(position * d)
    pe[:,1::2] = torch.cos(position * d)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
  def forward(self, x):
    return x + self.pe[:, :x.size(1), :]

class MyGPT(nn.Module):
  def __init__(self, d_model, n_embeddings, n_decoder_blocks, n_attention_heads=8, max_seq_len=512):
    super(MyGPT, self).__init__()
    self.embedding = nn.Embedding(n_embeddings, d_model)
    self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
    decoders = [DecoderBlock(d_model, n_attention_heads, max_seq_len) for _ in range(n_decoder_blocks)]
    self.decoders = nn.Sequential(*decoders)
    self.output_layer =nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, n_embeddings)
    )
  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = self.decoders(x)
    x = self.output_layer(x)
    return x
