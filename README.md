# Transformer Encoder — Forward Pass (From Scratch)

Forward pass de um Encoder (6 camadas) conforme "Attention Is All You Need" (Vaswani et al., 2017). Apenas Python 3, NumPy e pandas.

## Setup

```bash
pip install -r requirements.txt
```

## Executar

```bash
python -m src.main
```

(No Windows, se `python` não existir: `py -m src.main`.)

## Estrutura

```
src/
├── attention.py
├── embeddings.py
├── ffn.py
├── layernorm.py
└── main.py
```

## Entrega

```bash
git tag v1.0
git push origin v1.0
```
