import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        def relu(z): 
            return np.maximum(0, z)
        def relu_derivative(z):
            return (z > 0).astype(float)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        z1 = np.dot(W1, x) + b1 
        a1 = relu(z1)
        z2 = np.dot(W2, a1) + b2
        loss = np.mean((z2 - y_true) ** 2) 

        n = len(y_true)
        dL_dpred = (2/n) * (z2 - y_true)

        dw2 = np.outer(dL_dpred, a1)
        db2 = dL_dpred 

        da1 = np.dot(W2.T, dL_dpred)
        dz1 = da1 * relu_derivative(z1)

        dw1 = np.outer(dz1, x)
        db1 = dz1 

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dw1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dw2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }


        
