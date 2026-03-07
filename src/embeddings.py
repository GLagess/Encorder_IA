import numpy as np
import pandas as pd


def criar_vocabulario():
    vocab_dict = {"o": 0, "banco": 1, "bloqueou": 2, "cartao": 3, "<pad>": 4}
    df = pd.DataFrame(list(vocab_dict.items()), columns=["palavra", "id"])
    return vocab_dict, df


def frase_para_ids(frase: str, vocab: dict) -> list:
    tokens = frase.lower().strip().split()
    return [vocab.get(t, vocab.get("<pad>", 4)) for t in tokens]


def criar_matriz_embeddings(vocab_size: int, d_model: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((vocab_size, d_model)).astype(np.float64) * 0.1
    return np.random.randn(vocab_size, d_model).astype(np.float64) * 0.1


def obter_embeddings(ids: list, emb_matrix: np.ndarray) -> np.ndarray:
    vecs = emb_matrix[np.array(ids)]
    return vecs[np.newaxis, :, :]
