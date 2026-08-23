import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = model_dim // num_heads

        self.group_size = num_heads // num_kv_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        #B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        batch_size, seq_len, _ = x.shape

        q_proj_out = self.q_proj(x)
        k_proj_out = self.k_proj(x)
        v_proj_out = self.v_proj(x)

        q = q_proj_out.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_tensor = k_proj_out.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_tensor = v_proj_out.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.group_size > 1:
            k = torch.repeat_interleave(k_tensor, repeats=self.group_size, dim=1)
            v = torch.repeat_interleave(v_tensor, repeats=self.group_size, dim=1)
        else:
            k = k_tensor
            v = v_tensor

        scores = q @ torch.transpose(k, -2, -1) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))


        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ v

        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, seq_len, self.model_dim)

        output = self.output_proj(attn_out)

        return torch.round(output, decimals=4)
