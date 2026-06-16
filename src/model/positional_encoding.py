import math
import torch
import torch.nn as nn


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


class RotaryPositionEmbedding(nn.Module):
  def __init__(self, head_dim, max_seq_len, theta=10000.0):
    super(RotaryPositionEmbedding, self).__init__()
    assert head_dim % 2 == 0

    inv_freq = 1.0 / (
      theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    position = torch.arange(0, max_seq_len, 1).float()
    freqs = torch.outer(position, inv_freq)

    self.register_buffer(
      "cos",
      freqs.cos()[None, None, :, :],
      persistent=False,
    )
    self.register_buffer(
      "sin",
      freqs.sin()[None, None, :, :],
      persistent=False,
    )
  def forward(self, x):
    # x: [B, H, T, D]
    T = x.size(2)
    cos = self.cos[:, :, :T, :].to(dtype=x.dtype, device=x.device)
    sin = self.sin[:, :, :T, :].to(dtype=x.dtype, device=x.device)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out
