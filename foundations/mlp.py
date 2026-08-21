import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        current_activation = np.array(x, dtype=np.float64)
        num_layers = len(weights)

        for i in range(num_layers):
            W = np.array(weights[i])
            b = np.array(biases[i])
            z = np.dot(current_activation, W) + b
            if i < num_layers - 1:
                current_activation = np.maximum(0,z)
            else:
                current_activation = z
        return np.round(current_activation, 5)
