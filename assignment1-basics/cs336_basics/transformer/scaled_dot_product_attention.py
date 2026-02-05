import math
import torch
from cs336_basics.transformer.softmax import SoftMax

class ScaledDotProductAttention(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.softmax_layer = SoftMax()

  def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
              mask: torch.Tensor):
    # querys: (batch_size, ..., d_k)
    # keys: (batch_size, ..., seq_len, d_k)
    # values: (batch_size, ..., seq_len, d_v)
    # mask: (seq_len, seq_len)
    attention_score = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    attention_score = attention_score.masked_fill(~mask, float("-inf"))
    attention_probs = self.softmax_layer(attention_score, -1)
    return attention_probs @ V