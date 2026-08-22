import torch
import torch.nn as nn
import math
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return (torch.randn(fan_out, fan_in) * std).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        return (torch.randn(fan_out, fan_in) * std).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        # 1. లేయర్ డెమెన్షన్స్ సెట్ చేయడం
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        
        # 2. అన్ని వెయిట్స్ ని ముందుగానే జనరేట్ చేయడం (Order preservation కోసం)
        for i in range(num_layers):
            fan_in = dims[i]
            fan_out = dims[i+1]
            
            if init_type == 'kaiming':
                std = math.sqrt(2.0 / fan_in)
            elif init_type == 'xavier':
                std = math.sqrt(2.0 / (fan_in + fan_out))
            else:
                std = 1.0 # random
                
            w = torch.randn(fan_out, fan_in) * std
            weights.append(w)
            
        # 3. ఇన్‌పుట్ వెక్టర్ క్రియేట్ చేయడం
        x = torch.randn(1, input_dim)
        stds = []
        
        # 4. ఫోర్వర్డ్ పాస్ చేయడం
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(float(x.std().item()), 2))
            
        return stds