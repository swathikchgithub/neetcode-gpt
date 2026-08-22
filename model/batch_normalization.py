import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        X = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        beta = np.array(beta, dtype=np.float64)
        r_mean = np.array(running_mean, dtype=np.float64)
        r_var = np.array(running_var, dtype=np.float64)
        
        if training:
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            
            x_hat = (X - batch_mean) / np.sqrt(batch_var + eps)
            
            r_mean = (1 - momentum) * r_mean + momentum * batch_mean
            r_var = (1 - momentum) * r_var + momentum * batch_var
        else:
            x_hat = (X - r_mean) / np.sqrt(r_var + eps)
            
        out = gamma * x_hat + beta
        
        # రౌండ్ చేసి లిస్ట్ రూపంలో పంపడం
        return (
            np.round(out, 4).tolist(),
            np.round(r_mean, 4).tolist(),
            np.round(r_var, 4).tolist()
        )

