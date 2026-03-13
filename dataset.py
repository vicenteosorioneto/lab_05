"""
dataset.py  —  Tarefa 1
───────────────────────
Carrega o dataset Helsinki-NLP/opus_books (en-pt) do Hugging Face e
retorna as primeiras n_samples frases como pares (src, tgt).

Se o dataset não estiver acessível (sem internet ou sem a biblioteca
`datasets` instalada), um corpus sintético de demonstração é usado para
garantir que o restante do laboratório possa ser executado.
"""

from __future__ import annotations

from typing import List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────────

def load_translation_pairs(
    src_lang: str = "en",
    tgt_lang: str = "pt",
    n_samples: int = 1000,
) -> List[Tuple[str, str]]:
    """
    Retorna até n_samples pares (frase_src, frase_tgt) do opus_books.

    Parâmetros
    ----------
    src_lang  : código do idioma de origem  (padrão 'en').
    tgt_lang  : código do idioma de destino (padrão 'pt').
    n_samples : quantidade máxima de pares  (padrão 1 000).
    """
    try:
        from datasets import load_dataset  # type: ignore

        print(
            f"[dataset] Carregando Helsinki-NLP/opus_books "
            f"({src_lang}-{tgt_lang}, até {n_samples} amostras)..."
        )
        dataset = load_dataset(
            "Helsinki-NLP/opus_books",
            f"{src_lang}-{tgt_lang}",
            split="train",
            trust_remote_code=True,
        )
        n = min(n_samples, len(dataset))
        pairs = [
            (ex["translation"][src_lang], ex["translation"][tgt_lang])
            for ex in dataset.select(range(n))
        ]
        print(f"[dataset] {len(pairs)} pares carregados com sucesso.\n")
        return pairs

    except Exception as exc:
        print(f"[dataset] Aviso: não foi possível carregar o dataset ({exc}).")
        print("[dataset] Usando corpus sintético de demonstração.\n")
        return _synthetic_pairs()[:n_samples]


# ──────────────────────────────────────────────────────────────────────────────
# Corpus sintético de fallback
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_pairs() -> List[Tuple[str, str]]:
    """
    50 pares EN→PT únicos repetidos 20 vezes = 1 000 amostras.
    A repetição maximiza a facilidade de overfitting para a Tarefa 4.
    """
    unique = [
        ("the cat is on the mat", "o gato está no tapete"),
        ("the dog runs in the park", "o cachorro corre no parque"),
        ("i love programming", "eu amo programação"),
        ("the machine learns well", "a máquina aprende bem"),
        ("thinking machines are fascinating", "máquinas pensantes são fascinantes"),
        ("the transformer is a neural network", "o transformer é uma rede neural"),
        ("attention is all you need", "atenção é tudo que você precisa"),
        ("the model trains on data", "o modelo treina nos dados"),
        ("deep learning is powerful", "aprendizado profundo é poderoso"),
        ("a quick brown fox jumps", "uma raposa marrom veloz pula"),
        ("the sun rises in the east", "o sol nasce no leste"),
        ("she reads a good book", "ela lê um bom livro"),
        ("we study artificial intelligence", "estudamos inteligência artificial"),
        ("the encoder reads the source sentence", "o codificador lê a frase de origem"),
        ("the decoder generates the target sentence", "o decodificador gera a frase de destino"),
        ("training loss must decrease over epochs", "a perda de treino deve diminuir ao longo das épocas"),
        ("the sky is blue today", "o céu está azul hoje"),
        ("natural language processing is fun", "processamento de linguagem natural é divertido"),
        ("python is a great programming language", "python é uma ótima linguagem de programação"),
        ("the weights are updated by gradient descent", "os pesos são atualizados pela descida do gradiente"),
        ("the forward pass computes predictions", "a passagem direta calcula as previsões"),
        ("the backward pass computes gradients", "a passagem inversa calcula os gradientes"),
        ("adam is an adaptive optimizer", "adam é um otimizador adaptativo"),
        ("cross entropy measures classification loss", "a entropia cruzada mede a perda de classificação"),
        ("padding tokens are ignored in the loss", "tokens de padding são ignorados na perda"),
        ("the causal mask prevents future leakage", "a máscara causal previne vazamento do futuro"),
        ("multi head attention captures context", "atenção multi-cabeça captura o contexto"),
        ("positional encoding adds order information", "a codificação posicional adiciona informação de ordem"),
        ("layer normalization stabilizes training", "a normalização de camada estabiliza o treinamento"),
        ("residual connections help gradients flow", "conexões residuais ajudam os gradientes a fluir"),
        ("the vocabulary maps words to integers", "o vocabulário mapeia palavras em inteiros"),
        ("embeddings are learned representations", "embeddings são representações aprendidas"),
        ("softmax converts logits to probabilities", "softmax converte logits em probabilidades"),
        ("batching speeds up training significantly", "o agrupamento em lotes acelera o treinamento"),
        ("overfitting means memorizing training data", "sobreajuste significa memorizar os dados de treino"),
        ("the model parameters are the learned weights", "os parâmetros do modelo são os pesos aprendidos"),
        ("matrix multiplication is the core operation", "a multiplicação de matrizes é a operação central"),
        ("relu is a simple nonlinear activation", "relu é uma ativação não linear simples"),
        ("the feed forward network expands dimensions", "a rede feed forward expande dimensões"),
        ("sequences are padded to equal length", "sequências são preenchidas para ter o mesmo comprimento"),
        ("the teacher forces correct tokens during training", "o professor força tokens corretos durante o treino"),
        ("autoregressive generation produces one token at a time", "a geração autorregressiva produz um token por vez"),
        ("the start token initiates decoding", "o token de início inicia a decodificação"),
        ("the end token terminates the sequence", "o token de fim encerra a sequência"),
        ("hugging face provides pretrained models", "hugging face fornece modelos pré-treinados"),
        ("tokenization converts text to numbers", "a tokenização converte texto em números"),
        ("the learning rate controls step size", "a taxa de aprendizado controla o tamanho do passo"),
        ("gradient clipping prevents exploding gradients", "o corte de gradiente previne gradientes explosivos"),
        ("the loss converges during successful training", "a perda converge durante um treinamento bem-sucedido"),
        ("neural networks learn hierarchical features", "redes neurais aprendem características hierárquicas"),
    ]
    # Repete 20x para atingir ~1000 pares
    return unique * 20
