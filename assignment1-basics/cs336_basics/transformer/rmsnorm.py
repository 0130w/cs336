import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) # TODO:

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_type = x.dtype
        # upcast input to torch.float32 to prevent overflow while squaring
        x = x.to(torch.float32)
        RMS_res = torch.sqrt(torch.sum(torch.square(x), dim=2, keepdim=True) / self.d_model + self.eps) # (batch_size, sequence_length)
        result = x * self.weights / RMS_res
        return result.to(input_type)