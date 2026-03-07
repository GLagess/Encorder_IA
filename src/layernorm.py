import numpy as np

EPSILON = 1e-6


def layer_norm(x: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True) + epsilon
    return (x - mean) / np.sqrt(var)
