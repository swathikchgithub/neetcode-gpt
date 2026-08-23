import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None or self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim =1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim =1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Apply a causal mask offset by the number of previously cached tokens
        # 6. Return (rounded output, kv_cache)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        new_k = self.k_proj(x)
        new_v = self.v_proj(x)

        if kv_cache is None:
            kv_cache = KVCache()

        k, v = kv_cache.update(new_k, new_v)

        total_seq_len = k.shape[1]
        scores = q @ torch.transpose(k, 1, 2) / (k.shape[-1] ** 0.5)

        if total_seq_len > 1:
            mask = torch.full((seq_len, total_seq_len), float('-inf'), device=x.device)
            current_indices = torch.arange(total_seq_len - seq_len, total_seq_len, device=x.device).unsqueeze(1)
            all_indices = torch.arange(total_seq_len, device=x.device).unsqueeze(0)
            causal_mask = all_indices > current_indices
            scores = scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ v
        
        return torch.round(attn_out, decimals=4), kv_cache
