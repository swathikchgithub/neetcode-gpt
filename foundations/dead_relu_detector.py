import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        activations = {}
        hooks = []
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        for name, layer in model.named_modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(
                    layer.register_forward_hook(
                        get_activation(name)
                    )
                )
        with torch.no_grad():
            _ = model(x)
        
        for h in hooks:
            h.remove()

        for name, act in activations.items():
            if act.ndim == 2:
                dead_neurons = (act <= 0).all(dim=0)
                dead_fraction = float(
                    dead_neurons.float().mean().item()
                )
            else:
                dead_fraction = float(
                    (act <= 0).float().mean().item()
                )
            dead_fractions.append(
                round(dead_fraction, 4)
            )
        return dead_fractions


    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if not dead_fractions:
            return 'healthy'

        max_dead = max(dead_fractions)

        if max_dead > 0.5:
            return 'use_leaky_relu'
        
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
        
        is_increasing = all(
            dead_fractions[i] <= dead_fractions[i+1]
            for i in range(len(dead_fractions) - 1)
        )
        if is_increasing and dead_fractions[-1] > 0.1 and len(dead_fractions) > 1:
            return 'reduce_learning_rate'

        if max_dead < 0.1:
            return 'healthy'
        
        return 'healthy'
