import math
import torch

class Linear(torch.nn.Module):
  """ Linear layer not include a bias term, following most modern LLMs
  """
  def __init__(self, in_features: int, out_features: int,
               device : torch.device | None = None,
               dtype : torch.dtype | None = None):
    super().__init__()
    weight_data = torch.empty((out_features, in_features), device=device, dtype=dtype)
    mean = 0
    std = math.sqrt(2.0 / (in_features + out_features))
    # use trunc_normal_ instead of normal_ to avoid extreme value
    torch.nn.init.trunc_normal_(weight_data, mean=mean, std=std, a=-3 * std, b=3 * std)
    self.weights = torch.nn.Parameter(weight_data)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.weights.t()