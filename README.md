# Laboratório Técnico 05 — Treinamento Fim-a-Fim do Transformer

## Objetivo

Este laboratório encerra a Unidade I conectando o Transformer construído
nos Labs anteriores a um dataset real do Hugging Face e implementando o
Training Loop completo (Forward → Loss → Backward → Step).

O objetivo **não** é produzir um tradutor perfeito, mas sim:

1. Provar que a arquitetura consegue **aprender** — a curva de Loss deve
   cair significativamente ao longo das épocas.
2. Demonstrar que os **gradientes fluem corretamente** pelo modelo via
   *Overfitting Test* (Tarefa 4).

---

## Estrutura do projeto

```
lab_05/
├── transformer_from_scratch.py   # Modelo do Lab 04 (reutilizado)
├── dataset.py                    # Tarefa 1 — carregamento do dataset
├── tokenizer_utils.py            # Tarefa 2 — tokenização e vocabulário
├── train.py                      # Tarefa 3 — training loop
├── inference.py                  # Tarefa 4 — loop auto-regressivo
├── main.py                       # Ponto de entrada principal
└── requirements.txt
```

---

## Tarefas implementadas

### Tarefa 1 — Dataset Real (`dataset.py`)

- Carrega o dataset **Helsinki-NLP/opus_books** (`en-pt`) via `datasets`
  do Hugging Face.
- Seleciona as primeiras **1 000 frases** para garantir execução rápida
  em CPU / Google Colab gratuito.
- Inclui corpus sintético EN→PT como fallback automático caso o dataset
  esteja inacessível.

### Tarefa 2 — Tokenização Básica (`tokenizer_utils.py`)

- Usa `AutoTokenizer.from_pretrained("bert-base-multilingual-cased")`
  para converter frases em listas de inteiros.
- Constrói um **vocabulário local compacto** com apenas os token IDs que
  efetivamente aparecem no subconjunto de 1 000 frases.
- Adiciona `<START>` (ID 1) e `<EOS>` (ID 2) nas sequências de destino
  (Decoder).
- Aplica **padding** com ID 0 para igualar comprimentos no mini-batch.

### Tarefa 3 — Motor de Otimização (`train.py`)

- Instancia o `Transformer` do Lab 04 com `d_model=128 | h=4 | N=2`.
- `CrossEntropyLoss` com `ignore_index=0` — sem penalidade em padding.
- Otimizador **Adam** (mesmo do paper original).
- Loop de **15 épocas** com *teacher forcing*:
  - Encoder recebe os tokens de origem.
  - Decoder recebe a sequência alvo deslocada 1 posição à direita.
  - Loss = CrossEntropy(logits vs. tokens reais esperados).
  - `loss.backward()` + `optimizer.step()` atualizam WQ, WK, WV, WO…
  - Gradient clipping (`max_norm=1.0`) para estabilidade.

### Tarefa 4 — Prova de Fogo (`inference.py`)

- Pega frases que estiveram no mini-conjunto de treino.
- Usa o **loop auto-regressivo** do Lab 04 para gerar a tradução.
- O modelo deve reproduzir (ou se aproximar de) a tradução exata,
  comprovando que a arquitetura memorizou os padrões matriciais.

---

## Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar o laboratório completo
python main.py
```

Saída esperada:

```
TAREFA 1 — ...  [dataset carregado]
TAREFA 2 — ...  [vocab: ~N tokens]
TAREFA 3 — ...
  Época   Loss médio   Δ Loss
      1       4.2xxx        —
      2       3.8xxx   -0.3xxx
      ...
     15       1.2xxx   -0.0xxx    ← Loss caiu significativamente

TAREFA 4 — Prova de Fogo
  Entrada  : the cat is on the mat
  Esperado : o gato está no tapete
  Predito  : o gato está no tapete   ← memorização confirmada ✓
```

---

## Dependências

| Biblioteca     | Versão mínima | Papel                                  |
|----------------|---------------|----------------------------------------|
| `torch`        | 2.0           | Modelo, tensores, backpropagation      |
| `transformers` | 4.35          | AutoTokenizer (Tarefa 2)               |
| `datasets`     | 2.14          | Carregamento do opus_books (Tarefa 1)  |
| `numpy`        | 1.24          | Suporte numérico no modelo             |

---

## Nota sobre uso de IA Generativa

Todos os scripts deste laboratório (`dataset.py`, `tokenizer_utils.py`,
`train.py`, `inference.py`, `main.py`) foram **desenvolvidos em conjunto
com IA Generativa (GitHub Copilot / Claude)** e revisados por
[Vicente Osório Neto].

Detalhamento por tarefa:

- **Tarefa 1 — `dataset.py`**: carregamento e filtragem do dataset —
  escrito com IA, revisado por Vicente.
- **Tarefa 2 — `tokenizer_utils.py`**: tokenização com
  `bert-base-multilingual-cased` e padding — escrito com IA, revisado
  por Vicente.
- **Tarefa 3 — `train.py`**: o fluxo Forward / Loss / Backward / Step
  interage **estritamente** com as classes `Transformer` e
  `make_causal_mask` construídas por Vicente nos Labs 03 e 04 — escrito
  em conjunto com IA, revisado e validado por Vicente.
- **Tarefa 4 — `inference.py`**: loop auto-regressivo adaptado do Lab 04
  para o vocabulário real — escrito com IA, revisado por Vicente.

A arquitetura base (`transformer_from_scratch.py`) é a mesma entregue
no Lab 04, de autoria de Vicente, sem alterações.

---

## Versionamento

```bash
git tag v1.0
git push origin master --tags
```
