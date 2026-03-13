"""
transformer_from_scratch.py
───────────────────────────
Modelo Transformer Encoder-Decoder construído do zero com PyTorch.
Reutilizado do Laboratório 04 como exigido pelo enunciado do Lab 05.

Componentes:
  • scaled_dot_product_attention  – atenção com suporte a NumPy/Tensor
  • MultiHeadAttention             – WQ, WK, WV, WO by nn.Linear
  • PositionwiseFeedForward        – expansão com ReLU
  • AddNorm                        – conexão residual + LayerNorm
  • EncoderBlock / DecoderBlock    – bloco empilhável
  • PositionalEncoding             – PE senoidal fixo
  • Encoder / Decoder              – pilhas + embedding + projeção
  • Transformer                    – modelo completo
  • make_causal_mask               – máscara triangular inferior (look-ahead)
"""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

TensorLike = Union[torch.Tensor, np.ndarray]


# ──────────────────────────────────────────────────────────────────────────────
# Utilitário
# ──────────────────────────────────────────────────────────────────────────────

def to_torch(
    x: TensorLike,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    else:
        tensor = x
    if tensor.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.bool,
    ):
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Atenção
# ──────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
    mask: Optional[TensorLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_t = to_torch(q)
    k_t = to_torch(k)
    v_t = to_torch(v)

    dk = q_t.size(-1)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(dk)

    if mask is not None:
        mask_t = to_torch(mask, device=scores.device)
        if mask_t.dtype == torch.bool:
            scores = scores.masked_fill(~mask_t, float("-inf"))
        else:
            scores = scores + mask_t

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v_t)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisível por num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        attention_output = self._merge_heads(attention_output)
        output = self.w_o(attention_output)
        return output, attention_weights


# ──────────────────────────────────────────────────────────────────────────────
# Feed-Forward e Add & Norm
# ──────────────────────────────────────────────────────────────────────────────

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer_output))


# ──────────────────────────────────────────────────────────────────────────────
# Blocos Encoder / Decoder
# ──────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attention(x, x, x, src_mask)
        x = self.add_norm_1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.add_norm_2(x, ffn_out)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm_2 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.add_norm_3 = AddNorm(d_model, dropout)

    def forward(
        self,
        y: torch.Tensor,
        memory_z: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        masked_attn_out, _ = self.masked_self_attention(y, y, y, tgt_mask)
        y = self.add_norm_1(y, masked_attn_out)
        cross_attn_out, _ = self.cross_attention(y, memory_z, memory_z, memory_mask)
        y = self.add_norm_2(y, cross_attn_out)
        ffn_out = self.ffn(y)
        y = self.add_norm_3(y, ffn_out)
        return y


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ──────────────────────────────────────────────────────────────────────────────
# Encoder e Decoder completos
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(src_tokens)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory_z: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.embedding(tgt_tokens)
        y = self.positional_encoding(y)
        for layer in self.layers:
            y = layer(y, memory_z, tgt_mask, memory_mask)
        logits = self.output_projection(y)
        probs = self.softmax(logits)
        return logits, probs


# ──────────────────────────────────────────────────────────────────────────────
# Transformer completo
# ──────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory_z = self.encoder(src_tokens, src_mask)
        logits, probs = self.decoder(tgt_tokens, memory_z, tgt_mask, memory_mask)
        return logits, probs


# ──────────────────────────────────────────────────────────────────────────────
# Máscara causal (look-ahead)
# ──────────────────────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Retorna máscara triangular inferior booleana: shape [1, 1, seq_len, seq_len].
    Valor True = posição acessível; False = posição futura (bloqueada).
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Demo herdada do Lab 04 (execução standalone)
# ──────────────────────────────────────────────────────────────────────────────

def run_toy_inference() -> None:
    torch.manual_seed(7)

    src_vocab = {"<PAD>": 0, "Thinking": 1, "Machines": 2}
    tgt_vocab = {
        "<PAD>": 0, "<START>": 1, "<EOS>": 2,
        "Máquinas": 3, "Pensantes": 4, "Inteligentes": 5, "de": 6, "Teste": 7,
    }
    id_to_tgt = {idx: token for token, idx in tgt_vocab.items()}

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout=0.0,
    )
    model.eval()

    encoder_input = torch.tensor([[src_vocab["Thinking"], src_vocab["Machines"]]], dtype=torch.long)
    generated = [tgt_vocab["<START>"]]
    max_steps = 10
    eos_id = tgt_vocab["<EOS>"]

    print("Frase de entrada:", encoder_input.tolist())
    with torch.no_grad():
        while len(generated) < max_steps:
            decoder_input = torch.tensor([generated], dtype=torch.long)
            tgt_mask = make_causal_mask(decoder_input.size(1), device=decoder_input.device)
            logits, probs = model(encoder_input, decoder_input, tgt_mask=tgt_mask)
            next_token_id = int(torch.argmax(probs[:, -1, :], dim=-1).item())
            if len(generated) == max_steps - 1 and eos_id not in generated:
                next_token_id = eos_id
            generated.append(next_token_id)
            token = id_to_tgt[next_token_id]
            prob_val = float(probs[0, -1, next_token_id].item())
            print(f"Passo {len(generated)-1}: token='{token}' prob={prob_val:.4f}")
            if next_token_id == eos_id:
                break

    print("Sequência gerada:", [id_to_tgt[i] for i in generated])


if __name__ == "__main__":
    run_toy_inference()
