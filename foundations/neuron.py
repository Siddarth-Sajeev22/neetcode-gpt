import numpy as np
from numpy.typing import NDArray


class Solution:
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
        def sigmoid(z): 
            return 1 / (1 + np.exp(-z))
        def relu(z): 
            return np.maximum(0,z)
        z = np.dot(x, w) + b
        if activation.strip().lower() == "sigmoid": 
            z = sigmoid(z)
        else : 
            z = relu(z)
        return np.round(z, 5)


