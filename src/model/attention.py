import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.positional_encoding import RotaryPositionEmbedding

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


class MultiHeadAttentionV2(nn.Module):
  def __init__(self, d_model, n_heads, max_seq_len=512):
    super(MultiHeadAttentionV2, self).__init__()
    assert d_model % n_heads == 0
    assert (d_model // n_heads) % 2 == 0
    self.n_heads = n_heads
    self.pe = RotaryPositionEmbedding(d_model // n_heads, max_seq_len)
    self.w_qkv = nn.Linear(d_model, d_model * 3, bias=False)
    self.linear = nn.Linear(d_model, d_model, bias=False)
    self.max_seq_len = max_seq_len

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.w_qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(B, T, self.n_heads, C // self.n_heads)
    q = q.transpose(1,2)
    q = self.pe(q)

    k = k.view(B, T, self.n_heads, C // self.n_heads)
    k = k.transpose(1,2)
    k = self.pe(k)

    v = v.view(B, T, self.n_heads, C // self.n_heads)
    v = v.transpose(1,2)

    attention_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attention_output = attention_output.transpose(1,2).contiguous().view(B, T, C)
    return self.linear(attention_output)
