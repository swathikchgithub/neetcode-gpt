import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        self.heads = nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, self.head_dim) for _ in range(num_heads)
        ])
        self.out_linear = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run each head on the input, concatenate outputs along dim=2
        head_outputs = [head(embedded) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.out_linear(concatenated)
        return torch.round(output, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2)
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length, device=embedded.device))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v