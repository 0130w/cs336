import torch
from cs336_basics.transformer.linear import Linear

class GLU(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        # TODO: combine two kernel to one big kernel
        super().__init__()
        self.w1_linear_layer = Linear(in_features, out_features, device, dtype)
        self.w2_linear_layer = Linear(in_features, out_features, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w1_linear_layer(x)) * self.w2_linear_layer(x)