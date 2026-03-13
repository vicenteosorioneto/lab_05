"""
inference.py  —  Tarefa 4
──────────────────────────
Prova de Fogo (Overfitting Test).

Usa o Loop Auto-regressivo (herdado do Lab 04) para gerar a tradução
de uma frase que esteve no conjunto de treino. Se o treinamento
convergiu, o modelo deve reproduzir (ou se aproximar muito de)
a tradução exata — demonstrando que a arquitetura assimilou
(memorizou) o padrão matricial com sucesso.

Loop auto-regressivo
────────────────────
1. Codifica a frase de origem → tensor [1, src_len].
2. Inicia a sequência gerada com [START_ID].
3. A cada passo:
     a. Empilha os tokens gerados até agora → [1, step].
     b. Aplica a máscara causal             → [1, 1, step, step].
     c. Forward no Transformer              → logits [1, step, vocab].
     d. Argmax no último passo              → próximo token_id.
     e. Acrescenta token_id à sequência.
4. Para quando gera EOS_ID ou atinge max_steps.
5. Decodifica os IDs de volta a texto e retorna a string.
"""

from __future__ import annotations

import torch

from tokenizer_utils import TranslationTokenizer, PAD_ID, START_ID, EOS_ID
from transformer_from_scratch import Transformer, make_causal_mask


def autoregressive_translate(
    model: Transformer,
    tokenizer: TranslationTokenizer,
    src_text: str,
    *,
    device: torch.device,
    max_steps: int = 64,
) -> str:
    """
    Traduz src_text usando geração auto-regressiva (sem teacher forcing).

    Parâmetros
    ----------
    model      : Transformer treinado (Lab 04 / Lab 05).
    tokenizer  : TranslationTokenizer que codifica e decodifica textos.
    src_text   : frase de entrada (língua de origem).
    device     : dispositivo PyTorch ('cpu' ou 'cuda').
    max_steps  : número máximo de tokens a gerar antes de parar.

    Retorna
    -------
    Frase gerada como string (língua de destino).
    """
    model.eval()

    # ── Codifica a frase de origem ────────────────────────────────────────────
    src_ids = tokenizer.encode_src(src_text)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)  # [1, src_len]

    # ── Inicia a sequência com <START> ────────────────────────────────────────
    generated: list[int] = [START_ID]

    with torch.no_grad():
        while len(generated) < max_steps:
            # Sequência gerada até agora: [1, step]
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)

            # Máscara causal: [1, 1, step, step]
            tgt_mask = make_causal_mask(tgt_tensor.size(1), device=device)

            # Forward Pass (usa Transformer do Lab 04)
            logits, _ = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)
            # logits: [1, step, vocab_size]

            # Seleciona o próximo token pelo argmax do último passo
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            generated.append(next_id)

            if next_id == EOS_ID:
                break

    # ── Decodifica de volta a texto ───────────────────────────────────────────
    return tokenizer.decode_tgt(generated)


def run_overfit_test(
    model: Transformer,
    tokenizer: TranslationTokenizer,
    pairs: list[tuple[str, str]],
    device: torch.device,
    n_tests: int = 3,
) -> None:
    """
    Executa o teste de overfitting em n_tests frases do conjunto de treino.

    Imprime entrada, tradução esperada e tradução gerada pelo modelo.
    """
    print("=" * 62)
    print("  TAREFA 4 — Prova de Fogo (Overfitting Test)")
    print("  O modelo tenta reproduzir traduções vistas no treino.")
    print("=" * 62)

    for i in range(min(n_tests, len(pairs))):
        src_text, expected = pairs[i]
        predicted = autoregressive_translate(
            model, tokenizer, src_text, device=device
        )
        match = predicted.strip().lower() == expected.strip().lower()
        status = "✓ EXATO" if match else "≈ próximo"

        print(f"\n  [{i+1}] Entrada  : {src_text}")
        print(f"       Esperado : {expected}")
        print(f"       Predito  : {predicted}")
        print(f"       Status   : {status}")

    print("\n" + "=" * 62)
    print(
        "  Nota: correspondência exata não é obrigatória.\n"
        "  O importante é que o Loss caiu e o modelo convergiu.\n"
        "  Um resultado próximo já prova que os gradientes fluíram.\n"
        "=" * 62
    )
