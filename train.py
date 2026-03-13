"""
train.py  —  Tarefa 3
──────────────────────
Motor de Otimização: implementa o Training Loop completo com:

  • CrossEntropyLoss  (ignore_index=PAD_ID — sem penalidade em padding)
  • Otimizador Adam   (mesmo usado no paper original "Attention is All You Need")
  • Teacher forcing   (decoder recebe a sequência alvo deslocada 1 posição)
  • Gradient clipping (estabilidade numérica)
  • Máscara causal    (look-ahead mask via make_causal_mask do Lab 04)

O fluxo de Forward/Backward interage estritamente com as classes
construídas no Lab 04 (Transformer, make_causal_mask).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tokenizer_utils import PAD_ID
from transformer_from_scratch import Transformer, make_causal_mask


# ──────────────────────────────────────────────────────────────────────────────
# Preparação dos dados
# ──────────────────────────────────────────────────────────────────────────────

def _build_dataloader(
    src_ids: List[List[int]],
    tgt_ids: List[List[int]],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    src_t = torch.tensor(src_ids, dtype=torch.long)
    tgt_t = torch.tensor(tgt_ids, dtype=torch.long)
    dataset = TensorDataset(src_t, tgt_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: Transformer,
    src_ids: List[List[int]],
    tgt_ids: List[List[int]],
    *,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    pad_id: int = PAD_ID,
) -> Transformer:
    """
    Executa o treinamento completo e retorna o modelo treinado.

    Fluxo por mini-batch
    --------------------
    1. Encoder recebe os tokens da língua de origem.
    2. Decoder recebe os tokens da língua destino deslocados 1 posição
       à direita (Teacher Forcing):
         decoder_input = tgt[:, :-1]   → [<START>, t1, t2, …, tN-1]
         labels        = tgt[:, 1:]    → [t1, t2, …, tN-1, <EOS>]
    3. Loss = CrossEntropy(logits, labels) — ignora posições PAD.
    4. loss.backward() + optimizer.step() atualizam WQ, WK, WV, WO, …
    """
    model.to(device)
    model.train()

    # ── Função de perda: CrossEntropyLoss ────────────────────────────────────
    # ignore_index=pad_id → a rede não é penalizada por "errar" tokens de padding
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # ── Otimizador Adam ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = _build_dataloader(src_ids, tgt_ids, batch_size)

    print("=" * 58)
    print(f"  TREINAMENTO  |  épocas={epochs}  batch={batch_size}  lr={lr}")
    print(f"  dispositivo  : {device}")
    print(f"  amostras     : {len(src_ids)}")
    print("=" * 58)
    print(f"  {'Época':>5}  {'Loss médio':>12}  {'Δ Loss':>10}")
    print(f"  {'-'*32}")

    prev_loss = None

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)   # [B, src_len]
            tgt_batch = tgt_batch.to(device)   # [B, tgt_len]

            # ── Teacher Forcing ──────────────────────────────────────────
            # decoder_input: tgt sem o último token  → [B, tgt_len-1]
            # labels       : tgt sem o primeiro token → [B, tgt_len-1]
            decoder_input = tgt_batch[:, :-1]
            labels        = tgt_batch[:, 1:].contiguous()

            # ── Máscara causal para o Decoder ────────────────────────────
            tgt_seq_len = decoder_input.size(1)
            tgt_mask = make_causal_mask(tgt_seq_len, device=device)  # [1,1,T,T]

            # ── Forward Pass ─────────────────────────────────────────────
            # (usa estritamente a classe Transformer do Lab 04)
            logits, _ = model(src_batch, decoder_input, tgt_mask=tgt_mask)
            # logits: [B, tgt_len-1, vocab_size]

            # ── Cálculo da Loss ──────────────────────────────────────────
            vocab_size = logits.size(-1)
            # CrossEntropyLoss espera (N, C) → reshape para (B*T, vocab_size)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
            )

            # ── Backward Pass + Step ─────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()                                        # backprop
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # clipping
            optimizer.step()                                       # atualiza pesos

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        delta = f"{avg_loss - prev_loss:+.4f}" if prev_loss is not None else "    —"
        print(f"  {epoch:>5}  {avg_loss:>12.4f}  {delta:>10}")
        prev_loss = avg_loss

    print("=" * 58)
    print()
    return model
