import torch

class RoPE(torch.nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
    super().__init__()
    numerator = torch.arange(max_seq_len, dtype=torch.float)
    denominator = ( theta ** ( (torch.arange(0, d_k, 2).float()) / d_k) ).view(-1, 1).expand(-1, 2).reshape(-1)
    angle = numerator.unsqueeze(1) / denominator  # [max_sequence_len, d_model]
    self.register_buffer("sin_cached_table", angle.sin(), persistent=False)
    self.register_buffer("cos_cached_table", angle.cos(), persistent=False)

  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    x_reshaped = x.reshape(x.shape[:-1] + (-1, 2))
    x_even_idx, x_odd_idx = x_reshaped.unbind(dim=-1)
    # [batch_size, max_sequence_len, d_model]
    x_rotated = torch.stack((-x_odd_idx, x_even_idx), dim=-1).reshape(x.shape)
    # [max_sequence_len, d_model]
    cos_value = self.cos_cached_table[token_positions]  # type: ignore  
    sin_value = self.sin_cached_table[token_positions]  # type: ignore
    # [1, max_sequence_len, d_model]
    cos_value = cos_value.unsqueeze(0)
    sin_value = sin_value.unsqueeze(0)
    return x * cos_value + x_rotated * sin_value