import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        X = np.array(x, dtype=np.float64)
        Gamma = np.array(gamma, dtype=np.float64)

        rms = np.sqrt(np.mean( X ** 2) + eps)

        x_hat = X / rms

        out = x_hat * Gamma

        return np.round(out, 4).tolist()
