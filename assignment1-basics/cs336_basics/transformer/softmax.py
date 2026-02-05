import torch

class SoftMax(torch.nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = x.max(dim, keepdim=True)[0]
    x_exp = (x - max_val).exp()
    return x_exp / x_exp.sum(dim, keepdim=True)