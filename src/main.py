import numpy as np

from .embeddings import (
    criar_vocabulario,
    criar_matriz_embeddings,
    frase_para_ids,
    obter_embeddings,
)
from .attention import scaled_dot_product_attention
from .layernorm import layer_norm
from .ffn import ffn

D_MODEL = 64
D_K = 64
D_FF = 256
SEED = 42


def _init_attention_weights(d_model: int, d_k: int):
    Wq = np.random.randn(d_model, d_k).astype(np.float64) * 0.1
    Wk = np.random.randn(d_model, d_k).astype(np.float64) * 0.1
    Wv = np.random.randn(d_model, d_model).astype(np.float64) * 0.1
    return Wq, Wk, Wv


def _init_ffn_weights(d_model: int, d_ff: int):
    W1 = np.random.randn(d_model, d_ff).astype(np.float64) * 0.1
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model).astype(np.float64) * 0.1
    b2 = np.zeros(d_model)
    return W1, b1, W2, b2


def _encoder_layer(X, Wq, Wk, Wv, W1, b1, W2, b2, d_k):
    X_att = scaled_dot_product_attention(X, Wq, Wk, Wv, d_k)
    X_norm1 = layer_norm(X + X_att)
    X_ffn = ffn(X_norm1, W1, b1, W2, b2)
    X_out = layer_norm(X_norm1 + X_ffn)
    return X_out


def main():
    np.random.seed(SEED)

    vocab_dict, df_vocab = criar_vocabulario()
    print("Vocabulário (DataFrame):")
    print(df_vocab)
    print()

    frase = "o banco bloqueou cartao"
    ids = frase_para_ids(frase, vocab_dict)
    print(f"Frase: '{frase}'")
    print(f"IDs: {ids}")
    print()

    vocab_size = len(vocab_dict)
    emb_matrix = criar_matriz_embeddings(vocab_size, D_MODEL, seed=SEED)
    X = obter_embeddings(ids, emb_matrix)

    batch_size, seq_len, d_model = X.shape
    print(f"Tensor de entrada X: shape = {X.shape} (Batch, Tokens, d_model)")
    assert d_model == D_MODEL, f"d_model deve ser {D_MODEL}"
    print()

    num_layers = 6
    for camada in range(num_layers):
        Wq, Wk, Wv = _init_attention_weights(D_MODEL, D_K)
        W1, b1, W2, b2 = _init_ffn_weights(D_MODEL, D_FF)
        X = _encoder_layer(X, Wq, Wk, Wv, W1, b1, W2, b2, D_K)

    print(f"Tensor de saída Z (após 6 camadas): shape = {X.shape}")
    assert X.shape == (batch_size, seq_len, D_MODEL)
    print("Validação: entrada e saída com shape (Batch, Tokens, 64). OK.")


if __name__ == "__main__":
    main()
