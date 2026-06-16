import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
  def __init__(self, d_model, d_hidden=None):
    super(SwiGLUFFN, self).__init__()
    if d_hidden is None:
      d_hidden = self._calculate_d_hidden(d_model)
    self.w_gv = nn.Linear(d_model, d_hidden*2, bias=False)
    self.w_o = nn.Linear(d_hidden, d_model, bias=False)

  def _calculate_d_hidden(self, d_model, multiple_of = 64):
    d_hidden = int(8 * d_model / 3)
    return ((d_hidden + multiple_of - 1) // multiple_of) * multiple_of

  def forward(self, x):
    gate, value = self.w_gv(x).chunk(2, dim=-1)
    x = F.silu(gate) * value
    return self.w_o(x)