import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention, MultiHeadAttentionV2
from src.model.normalize import LayerNorm, RMSNorm
from src.model.layer import SwiGLUFFN

class DecoderBlock(nn.Module):
  def __init__(self, d_model, n_attention_heads, dropout=0.1, max_seq_len=512):
    super(DecoderBlock, self).__init__()
    self.normalize1 = LayerNorm(d_model)
    self.attention = MultiHeadAttention(d_model, n_attention_heads, max_seq_len)
    self.dropout1 = nn.Dropout(p=dropout)
    self.normalize2 = LayerNorm(d_model)
    self.ffn = nn.Sequential(
        nn.Linear(d_model, d_model*4),
        nn.GELU(),
        nn.Linear(d_model*4, d_model)
    )
    self.dropout2 = nn.Dropout(p=dropout)
  def forward(self, x):
    x = x + self.dropout1(self.attention(self.normalize1(x)))
    return x + self.dropout2(self.ffn(self.normalize2(x)))


class DecoderBlockV2(nn.Module):
  def __init__(self, d_model, n_attention_heads, dropout=0.1, max_seq_len=512):
    super(DecoderBlockV2, self).__init__()
    self.normalize1 = RMSNorm(d_model)
    self.attention = MultiHeadAttentionV2(d_model, n_attention_heads, max_seq_len)
    self.dropout1 = nn.Dropout(p=dropout)
    self.normalize2 = RMSNorm(d_model)
    self.ffn = SwiGLUFFN(d_model)
    self.dropout2 = nn.Dropout(p=dropout)

  def forward(self, x):
    x = x + self.dropout1(self.attention(self.normalize1(x)))
    x = x + self.dropout2(self.ffn(self.normalize2(x)))
    return x
