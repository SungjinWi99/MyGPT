import torch
import torch.nn as nn


class LayerNorm(nn.Module):
  def __init__(self, d_model, eps=1e-5):
    super(LayerNorm, self).__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.beta = nn.Parameter(torch.zeros(d_model))
  def forward(self, x):
    var, mean = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + self.eps)
    x = self.gamma * x + self.beta
    return x

class RMSNorm(nn.Module):
  def __init__(self, d_model, eps=1e-5):
    super(RMSNorm, self).__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(d_model))
  def forward(self, x):
    rms = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
    rms = torch.sqrt(rms + self.eps)
    return self.gamma * (x / rms)