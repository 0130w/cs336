import torch
from cs336_basics.transformer.silu import SiLU

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_ff is None:
            # TODO: optimize d_ff to ensure it's a multiple of 64
            d_ff = (int)(8 * d_model / 3)
        w1_weights_data = torch.ones((d_ff, d_model), device=device, dtype=dtype)
        w2_weights_data = torch.ones((d_model, d_ff), device=device, dtype=dtype)
        w3_weights_data = torch.ones((d_ff, d_model), device=device, dtype=dtype)
        self.silu_layer = SiLU()
        self.w1_weights = torch.nn.Parameter(w1_weights_data)
        self.w2_weights = torch.nn.Parameter(w2_weights_data)
        self.w3_weights = torch.nn.Parameter(w3_weights_data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.silu_layer(x @ self.w1_weights.t()) * (x @ self.w3_weights.t())) @ self.w2_weights.t()