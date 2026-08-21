import numpy as np
from numpy.typing import NDArray


class Solution:
    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        weighted_sum = np.dot(x, w) + b
        output = 0

        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-weighted_sum))

        elif activation == "relu":
            output = max(0.0, weighted_sum)

        else:
            output = weighted_sum

        return float(np.round(output, 5))
