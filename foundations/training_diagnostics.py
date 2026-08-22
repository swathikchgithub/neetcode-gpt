import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        activations = {}
        hooks = []
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear)):
                hooks.append(layer.register_forward_hook(get_activation(name)))
        
        with torch.no_grad():
            _ = model(x)

        for h in hooks: 
            h.remove()

        for name, act in activations.items():
            mean_val = float(act.mean().item())
            std_val = float(act.std().item()) if act.numel() > 1 else 0.0

            if act.ndim == 2:
                dead_neurons = (act <= 0).all(dim=0)
                dead_fraction = float(dead_neurons.float().mean().item())
            else:
                dead_fraction = float((act <= 0).float().mean().item())
            
            stats.append({
                'mean': round(mean_val, 4),
                'std': round(std_val, 4),
                'dead_fraction': round(dead_fraction, 4)
            })
        return stats

    



    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        criterion = nn.MSELoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        stats = []
        for name , layer, in model.named_modules():
            if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                grad = layer.weight.grad.detach()
                mean_val = float(grad.mean().item())
                std_val = float(grad.std().item()) if grad.numel() > 1 else 0.0
                norm_val = float(torch.norm(grad).item())

                stats.append({
                    'mean': round(mean_val, 4),
                    'std': round(std_val, 4),
                    'norm': round(norm_val, 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for stat in gradient_stats:
            if stat['norm'] > 1000.0:
                return 'exploding_gradients'

        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        # activation std checks
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            if stat['std'] > 10.0:
                return 'exploding_gradients'
        return 'healthy'
