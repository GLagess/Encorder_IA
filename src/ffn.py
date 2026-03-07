import numpy as np


def ffn(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    hidden = x @ W1 + b1
    relu_out = np.maximum(0, hidden)
    return relu_out @ W2 + b2
