import numpy as np

EPSILON = 1e-6


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)


def scaled_dot_product_attention(
    X: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    d_k: int,
) -> np.ndarray:
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)

    return np.matmul(attn_weights, V)
