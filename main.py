"""
main.py  —  Ponto de entrada
─────────────────────────────
Laboratório Técnico 05 — Treinamento Fim-a-Fim do Transformer

Executa as quatro tarefas do laboratório em sequência:

  Tarefa 1  Carregamento do dataset real (Hugging Face)
  Tarefa 2  Tokenização básica com vocabulário local compacto
  Tarefa 3  Training Loop (Forward → Loss → Backward → Step)
  Tarefa 4  Prova de Fogo — Overfitting Test auto-regressivo
"""

from __future__ import annotations

import torch

from dataset import load_translation_pairs
from tokenizer_utils import TranslationTokenizer
from transformer_from_scratch import Transformer
from train import train_model
from inference import run_overfit_test


# ──────────────────────────────────────────────────────────────────────────────
# Hiperparâmetros
# ──────────────────────────────────────────────────────────────────────────────

N_SAMPLES   = 1000   # subconjunto de frases do dataset
MAX_LEN     = 64     # comprimento máximo de sequência (tokens)
D_MODEL     = 128    # dimensão do modelo
NUM_HEADS   = 4      # cabeças de atenção
D_FF        = 256    # dimensão interna do Feed-Forward
NUM_LAYERS  = 2      # camadas Encoder e Decoder
DROPOUT     = 0.1    # taxa de dropout (desativado na inferência)
EPOCHS      = 15     # épocas de treinamento
BATCH_SIZE  = 16     # tamanho do mini-batch
LR          = 1e-3   # taxa de aprendizado Adam


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ──────────────────────────────────────────────────────────────────────────
    # Tarefa 1 — Dataset Real
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 62)
    print("  TAREFA 1 — Carregamento do Dataset (Hugging Face)")
    print("━" * 62)

    pairs = load_translation_pairs(n_samples=N_SAMPLES)

    print(f"  Exemplo de par carregado:")
    print(f"    EN: {pairs[0][0]}")
    print(f"    PT: {pairs[0][1]}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Tarefa 2 — Tokenização
    # ──────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  TAREFA 2 — Tokenização Básica")
    print("━" * 62)

    tokenizer = TranslationTokenizer(pairs, max_len=MAX_LEN)
    src_ids, tgt_ids = tokenizer.encode_corpus(pairs)

    print(f"  IDs de origem  (primeiros 10): {src_ids[0][:10]}")
    print(f"  IDs de destino (primeiros 10): {tgt_ids[0][:10]}")
    print(f"  (ID 0=PAD  |  ID 1=<START>  |  ID 2=<EOS>)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Tarefa 3 — Training Loop
    # ──────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    print("  TAREFA 3 — Training Loop (Forward / Loss / Backward / Step)")
    print("━" * 62 + "\n")

    vocab_size = tokenizer.vocab_size

    # Instância o Transformer (construído nos Labs anteriores)
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parâmetros treináveis : {total_params:,}")
    print(f"  Vocabulário local     : {vocab_size} tokens\n")

    model = train_model(
        model,
        src_ids,
        tgt_ids,
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Tarefa 4 — Prova de Fogo
    # ──────────────────────────────────────────────────────────────────────────
    print("━" * 62)
    run_overfit_test(model, tokenizer, pairs, device=device, n_tests=3)


if __name__ == "__main__":
    main()
