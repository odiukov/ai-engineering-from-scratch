# Multi-Head Attention

> One attention head learns one relation at a time. Eight heads learn eight. Heads are free. Take more of them.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## The Problem

A single self-attention head computes one attention matrix. That matrix captures one kind of relationship — usually the one that minimizes loss on whatever the training signal is. If your data has subject-verb agreement, co-reference, long-range discourse, and syntactic chunking all tangled together, a single head smears them into a single soft-max distribution and loses half the signal.

The fix from the 2017 Vaswani paper: run several attention functions in parallel, each with its own Q, K, V projections, and concatenate the outputs. Each head operates in a smaller subspace of dimension `d_model / n_heads`. Total parameters stay the same. Expressive power goes up.

Multi-head attention is the default every transformer in 2026 ships with. The only argument is about *how many* heads and whether keys and values share projections (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## The Concept

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.** Take `X` of shape `(N, d_model)`. Project to Q, K, V each of shape `(N, d_model)`. Reshape to `(N, n_heads, d_head)` where `d_head = d_model / n_heads`. Transpose to `(n_heads, N, d_head)`.

**Attend in parallel.** Run scaled dot-product attention inside each head. Each head produces `(N, d_head)`. The heads operate on different subspaces of the embedding and never talk during the attention computation itself.

**Concatenate and project.** Stack heads back to `(N, d_model)` and multiply by a learned output matrix `W_o` of shape `(d_model, d_model)`. `W_o` is where heads get to mix.

**Why it works.** Each head can specialize without competing with the others for representational budget. Probing studies from 2019–2024 show distinct head roles: positional heads, head that attends to the previous token, copy heads, named-entity heads, induction heads (which underlie in-context learning).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA is the modern default because it cuts KV-cache memory by a factor of `N/G` while keeping nearly full quality. MLA goes further by compressing K/V into a latent space, then projecting back at compute time — costs FLOPs, saves a lot more memory.

```figure
multihead-split
```

## Build It

### Step 1: split heads from the single-head attention we already have

Lesson 02's single-head attention lives in `02-self-attention-from-scratch/code/self_attention.py` and is written on numpy. Do not import it here: this lesson's `code/main.py` is pure stdlib — no numpy, no torch — so it re-derives the same scaled dot-product attention on a tiny row-major `Matrix` class (`get`, `set`, `row`) and stays runnable on a bare Python install. Same math, one file, zero dependencies.

So take Lesson 02's attention as the *formula* you already know, and wrap it in a split/concat pair. Splitting is just slicing a contiguous band of columns per head:

```python
def split_heads(X: Matrix, n_heads: int) -> List[Matrix]:
    assert X.cols % n_heads == 0, "d_model not divisible by n_heads"
    d_head = X.cols // n_heads
    heads = []
    for h in range(n_heads):
        H = Matrix(X.rows, d_head)
        for i in range(X.rows):
            for j in range(d_head):
                H.set(i, j, X.get(i, h * d_head + j))
        heads.append(H)
    return heads

def combine_heads(heads: List[Matrix]) -> Matrix:
    n, d_head = heads[0].rows, heads[0].cols
    out = Matrix(n, d_head * len(heads))
    for h, H in enumerate(heads):
        for i in range(n):
            for j in range(d_head):
                out.set(i, h * d_head + j, H.get(i, j))
    return out
```

With numpy or torch this is one `reshape` plus one `transpose` and no Python loop — the head axis is a view, not a copy. That is exactly what `nn.MultiheadAttention` does under the hood; here we spell the indexing out so you can see which columns belong to which head.

### Step 2: run scaled-dot-product attention per head

Each head gets its own slice of Q, K, V. In stdlib that is a loop over heads:

```python
def multi_head_attention(X: Matrix, Wq, Wk, Wv, Wo, n_heads: int):
    Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)
    Qh = split_heads(Q, n_heads)          # list of (n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    head_outs = []
    per_head_weights = []
    for q, k, v in zip(Qh, Kh, Vh):
        o, w = scaled_dot_product_attention(q, k, v)   # Lesson 02's math, on Matrix
        head_outs.append(o)
        per_head_weights.append(w)
    concat = combine_heads(head_outs)
    return matmul(concat, Wo), per_head_weights
```

`scaled_dot_product_attention` is Lesson 02's function with numpy swapped for `Matrix` ops — `softmax_rows(matmul(q, transpose(k)) * (1/math.sqrt(q.cols)))`, then `matmul(weights, v)` — and it is defined in this lesson's `code/main.py`, not imported.

Our Python loop over heads is a teaching device. On real hardware the same math is one `bmm`: the GPU sees a single batched matmul of shape `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Adding heads is free.

### Step 3: Grouped-Query Attention variant

Only the key and value projections change. Q gets `n_heads` groups; K and V get `n_kv_heads < n_heads` groups and are repeated to match:

```python
def gqa_project(X: Matrix, W, n_kv_heads: int, n_heads: int) -> List[Matrix]:
    kv_small = split_heads(matmul(X, W), n_kv_heads)   # n_kv_heads matrices
    repeat = n_heads // n_kv_heads
    return [kv_small[i // repeat] for i in range(n_heads)]
```

Note the returned list holds *references*, not copies — `Kh[0]` and `Kh[1]` are the same object when `repeat = 2`. That is the whole point: one stored KV head serves several query heads.

At inference this saves memory because only `n_kv_heads` copies live in the KV cache, not `n_heads`. Llama 3 70B uses 64 query heads with 8 KV heads — an 8× cache shrink.

### Step 4: probe what each head learned

Run MHA on a short sentence with 4 heads. For each head, print the `(N, N)` attention matrix. You'll see different heads pick out different structure even with random initialization — that's partly signal, partly rotational symmetry in the subspaces.

## Use It

In PyTorch, the one-line version:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA as of PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?** Rules of thumb from production models in 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` almost always lands at 64 or 128. It is the unit of how much one head can "see." Drop below 32 and heads start fighting the scaling factor `sqrt(d_head)`; go above 256 and you lose the "many small specialists" benefit.

## Ship It

See `outputs/skill-mha-configurator.md`. The skill recommends head count, kv-head count, and projection strategy for a new transformer given parameter budget, sequence length, and deployment target.

## Exercises

1. **Easy.** Take the MHA from `code/main.py` and change `n_heads` from 1 to 16 with `d_model=64` fixed. Plot the loss of a tiny one-layer model on a synthetic copy task. Do more heads help, plateau, or hurt?
2. **Medium.** Implement MQA (one KV head shared across all query heads). Measure how much parameter count drops vs full MHA. Compute how much the KV-cache size shrinks at inference for N=2048.
3. **Hard.** Implement a tiny version of Multi-head Latent Attention: compress K,V to a rank-`r` latent, store the latent in the KV cache, decompress at attention time. At what `r` does cache memory cross below 1/8 of full MHA while quality stays within 1 bit of validation ppl?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Further Reading

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — the original multi-head spec.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — the MQA paper.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — how to convert MHA to GQA after training.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA and why it beats MHA/GQA on cache memory.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — mechanistic look at what heads actually do.
