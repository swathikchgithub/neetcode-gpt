import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        final_loss = 0.0

        for epoch in range(epochs):
            # 2. ప్రతి ఎపోక్‌కి సీడ్ సెట్ చేయడం
            torch.manual_seed(epoch)
            
            # 3. రాండమ్ స్టార్ట్ ఇండెక్స్‌ల నుండి X మరియు Y బ్యాచ్‌లు తయారు చేయడం
            max_start = len(data) - context_length
            if max_start <= 0:
                raise ValueError("Data length must be greater than context_length")
                
            ix = torch.randint(0, max_start, (batch_size,))
            
            X = torch.stack([data[i : i + context_length] for i in ix])
            Y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])
            
            # 4. ఫార్వర్డ్ పాస్ (Forward Pass)
            optimizer.zero_grad()
            logits = model(X)  # Shape: (batch_size, context_length, vocab_size)
            
            # 5. క్రాస్-ఎంట్రోపీ కోసం టెన్సర్‌లను ఫ్లాటెన్ చేయడం ((B*T, C) మరియు (B*T))
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = Y.view(B * T)
            
            # 6. లాస్ లెక్కించడం మరియు బ్యాక్‌ప్రొపగేషన్
            loss = F.cross_entropy(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            
        return round(float(final_loss), 4)


