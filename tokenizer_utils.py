"""
tokenizer_utils.py  —  Tarefa 2
────────────────────────────────
Usa AutoTokenizer (bert-base-multilingual-cased) para converter frases
em sequências de inteiros, construindo um vocabulário local compacto
extraído exclusivamente do corpus de 1 000 frases.

Tokens especiais fixos
──────────────────────
  PAD_ID   = 0   padding (sequências de comprimento variável)
  START_ID = 1   <START> — inserido no início de toda sequência alvo
  EOS_ID   = 2   <EOS>   — inserido no fim de toda sequência alvo
  UNK_ID   = 3   <UNK>   — token desconhecido (fallback)

Os demais IDs locais (4, 5, 6, …) são atribuídos sequencialmente aos
IDs do BERT que efetivamente aparecem no corpus, de forma determinística
(ordem crescente de ID BERT).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

# IDs especiais locais
PAD_ID   = 0
START_ID = 1
EOS_ID   = 2
UNK_ID   = 3
_RESERVED = 4   # próximo ID livre após os especiais


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizador simples de fallback (sem dependência de transformers)
# ──────────────────────────────────────────────────────────────────────────────

class _WordTokenizer:
    """Tokenizador whitespace-level usado se `transformers` não estiver disponível."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.cls_token_id = 1   # → START_ID
        self.sep_token_id = 2   # → EOS_ID
        self.unk_token_id = 3
        self._w2i: Dict[str, int] = {}
        self._next = _RESERVED  # começa depois dos 4 especiais

    def _get_or_add(self, word: str) -> int:
        if word not in self._w2i:
            self._w2i[word] = self._next
            self._next += 1
        return self._w2i[word]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        words = text.lower().split()
        ids = [self._get_or_add(w) for w in words]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        inv = {v: k for k, v in self._w2i.items()}
        tokens = [inv[i] for i in ids if i in inv]
        return " ".join(tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Classe principal
# ──────────────────────────────────────────────────────────────────────────────

class TranslationTokenizer:
    """
    Tokenizador de tradução com vocabulário compacto.

    Parâmetros
    ----------
    pairs      : lista de (src_text, tgt_text) do corpus de treino.
    model_name : nome do tokenizador HuggingFace (padrão bert-base-multilingual-cased).
    max_len    : comprimento máximo de sequência após padding.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        model_name: str = "bert-base-multilingual-cased",
        max_len: int = 64,
    ) -> None:
        self.max_len = max_len

        # ── tenta carregar tokenizador HuggingFace ────────────────────────
        try:
            from transformers import AutoTokenizer  # type: ignore

            print(f"[tokenizer] Carregando '{model_name}'...")
            self._hf = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            print(f"[tokenizer] Aviso: {exc}. Usando tokenizador whitespace simples.")
            self._hf = _WordTokenizer()

        # ── IDs especiais do backend ──────────────────────────────────────
        bert_pad = self._hf.pad_token_id   # 0   → PAD_ID
        bert_cls = self._hf.cls_token_id   # 101 → START_ID
        bert_sep = self._hf.sep_token_id   # 102 → EOS_ID
        bert_unk = self._hf.unk_token_id   # 100 → UNK_ID
        _special_bert = {bert_pad, bert_cls, bert_sep, bert_unk}

        # Mapeia especiais com IDs locais fixos
        self._bert_to_local: Dict[int, int] = {
            bert_pad: PAD_ID,
            bert_cls: START_ID,
            bert_sep: EOS_ID,
            bert_unk: UNK_ID,
        }

        # ── varre corpus para coletar todos os IDs BERT do corpus ─────────
        all_bert_ids: set[int] = set()
        for src, tgt in pairs:
            for text in (src, tgt):
                ids = self._hf.encode(
                    text,
                    add_special_tokens=False,
                    max_length=max_len,
                    truncation=True,
                )
                all_bert_ids.update(ids)

        # Remove os especiais já mapeados
        corpus_ids = sorted(all_bert_ids - _special_bert)

        # Atribui IDs locais sequenciais a partir de _RESERVED
        for local_id, bert_id in enumerate(corpus_ids, start=_RESERVED):
            self._bert_to_local[bert_id] = local_id

        # Tabela inversa: local_id → bert_id  (lista indexada por local_id)
        self._local_to_bert: List[int] = [0] * len(self._bert_to_local)
        for bert_id, local_id in self._bert_to_local.items():
            self._local_to_bert[local_id] = bert_id

        print(f"[tokenizer] Vocabulário local: {self.vocab_size} tokens.\n")

    # ── propriedade ───────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._bert_to_local)

    # ── codificação ───────────────────────────────────────────────────────────

    def encode_src(self, text: str) -> List[int]:
        """
        Codifica frase de origem → IDs locais + padding até max_len.
        Sem tokens especiais nas extremidades (só o conteúdo).
        """
        bert_ids = self._hf.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
        )
        local = [self._bert_to_local.get(bid, UNK_ID) for bid in bert_ids]
        return local + [PAD_ID] * (self.max_len - len(local))

    def encode_tgt(self, text: str) -> List[int]:
        """
        Codifica frase de destino → [START] + IDs locais + [EOS] + padding.

        O token <START> sinaliza o início da decodificação;
        o token <EOS> sinaliza o fim — ambos exigidos pelo enunciado.
        """
        bert_ids = self._hf.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_len - 2,   # reserva posições para START e EOS
            truncation=True,
        )
        local = [self._bert_to_local.get(bid, UNK_ID) for bid in bert_ids]
        full = [START_ID] + local + [EOS_ID]
        return full + [PAD_ID] * (self.max_len - len(full))

    def decode_tgt(self, local_ids: List[int]) -> str:
        """Converte lista de IDs locais de volta a texto (melhor esforço)."""
        bert_ids = []
        for lid in local_ids:
            if lid == PAD_ID or lid == START_ID:
                continue
            if lid == EOS_ID:
                break
            if lid < len(self._local_to_bert):
                bert_ids.append(self._local_to_bert[lid])
            else:
                bert_ids.append(self._hf.unk_token_id)
        return self._hf.decode(bert_ids, skip_special_tokens=True)

    # ── codificação em lote ───────────────────────────────────────────────────

    def encode_corpus(
        self, pairs: List[Tuple[str, str]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Codifica todos os pares do corpus.

        Retorna
        -------
        src_ids : lista de listas de IDs de origem (com padding).
        tgt_ids : lista de listas de IDs de destino (com START/EOS/padding).
        """
        src_all: List[List[int]] = []
        tgt_all: List[List[int]] = []
        for src, tgt in pairs:
            src_all.append(self.encode_src(src))
            tgt_all.append(self.encode_tgt(tgt))
        print(f"[tokenizer] {len(src_all)} pares codificados (max_len={self.max_len}).\n")
        return src_all, tgt_all
