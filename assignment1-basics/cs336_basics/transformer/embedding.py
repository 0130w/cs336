import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight_data = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        mean = 0
        std = 1
        torch.nn.init.trunc_normal_(weight_data, mean=mean, std=std, a = -3, b = 3)
        self.weights = torch.nn.Parameter(weight_data)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # advanced index
        return self.weights[token_ids]