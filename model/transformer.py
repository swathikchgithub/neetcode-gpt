import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, h_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.W_k = nn.Linear(d_model, h_dim, bias=False)
        self.W_q = nn.Linear(d_model, h_dim, bias=False)
        self.W_v = nn.Linear(d_model, h_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        K, Q, V = self.W_k(x), self.W_q(x), self.W_v(x)
        scale = K.shape[-1] ** 0.5
        
        
        attn_scores = (Q @ K.transpose(-2, -1)) / scale
        
        
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
        
        weights = F.softmax(attn_scores, dim=-1)
        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.heads = nn.ModuleList([
            SingleHeadAttention(model_dim, model_dim // num_heads) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        head_results = [head(x) for head in self.heads]
        concat_out = torch.cat(head_results, dim=-1)
        return self.proj(concat_out)


class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.attn_layer = MultiHeadAttention(model_dim, num_heads)
        self.ffn_layer = FeedForwardNetwork(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        
        h = embedded + self.attn_layer(self.norm1(embedded))
        out = h + self.ffn_layer(self.norm2(h))
        return torch.round(out, decimals=4)