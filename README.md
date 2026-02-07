# SGT v2 — Block-Sparse Transformer (from scratch)

Train a 150M parameter GPT with block-sparse FFNs from scratch, alongside
a dense baseline, and compare quality + speed.

## The idea

Instead of retrofitting sparsity onto a pre-trained model (SGT Phase 1–6 showed
this doesn't work), we train a model that learns *how* to be sparse from the start.

Each FFN layer has a lightweight router (hidden → 128 → num_blocks) that selects
the top-k blocks per token. Only selected blocks participate in the SwiGLU
computation. The router is trained jointly with a straight-through estimator,
plus an auxiliary load-balancing loss to prevent block collapse.

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden dim | 768 |
| Intermediate dim | 2048 |
| Attention heads | 12 |
| Block size | 64 |
| Num blocks | 32 |
| Top-k blocks | 16 (50% target sparsity) |
| Vocab | 50257 (GPT-2 tokenizer) |
| Total params | ~150M |
| Router overhead | ~0.3% of params |

## Quick start

```bash
# Install deps
pip install -r requirements.txt

# Train both dense and sparse (500M tokens, ~2-3 hours on RTX PRO 6000)
python train.py --dataset /path/to/fineweb-edu.jsonl

# Evaluate
python eval.py --dataset /path/to/fineweb-edu.jsonl

# Train sparse only, more tokens
python train.py --dataset /path/to/fineweb-edu.jsonl --model sparse --max_tokens 1000000000

# Adjust sparsity target (fewer active blocks = more sparse)
python train.py --dataset /path/to/fineweb-edu.jsonl --top_k_blocks 8  # 75% sparsity
```

## Output structure

```
outputs/
├── dense/
│   ├── checkpoint.pt
│   └── train_log.json
├── sparse/
│   ├── checkpoint.pt
│   └── train_log.json
├── training_summary.json
└── eval_results.json
```

## What we're measuring

1. **PPL parity**: Does the sparse model match the dense model's perplexity?
   (Target: within 2%)

2. **Training throughput**: How much overhead does the router add during training?
   (At this scale, expect slight overhead due to mask operations)

3. **Decode latency**: At inference, is the sparse model faster?
   (This is the key question — if yes at 150M, the architecture scales)

4. **Block utilisation**: Does the router learn diverse routing patterns, or
   does it collapse to always using the same blocks?
   (Load-balancing loss should prevent collapse)

5. **Learned sparsity structure**: After training, do the block selection
   patterns show layer-dependent structure (early layers sparse, late layers
   dense, or vice versa)?

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | ~350 | SGTModel (sparse) + DenseModel (baseline) |
| `train.py` | ~280 | Training loop with streaming data, side-by-side comparison |
| `eval.py` | ~250 | PPL + latency + block utilisation analysis |

## Key design decisions

- **Mask-and-multiply, not gather/scatter**: At 150M scale, the overhead of
  index operations exceeds the savings from skipping FLOPs. We multiply by a
  binary mask instead, which zeros gradients to inactive blocks. This proves
  the *learning* concept; a production kernel would use actual sparse ops.

- **Straight-through estimator**: The top-k selection is non-differentiable.
  We use sigmoid soft probabilities in the backward pass so the router can
  learn which blocks to activate.

- **Load-balancing loss**: Without it, the router collapses to always selecting
  the same blocks (rich-get-richer). The aux loss penalises deviation from
  uniform block utilisation.

- **GPT-2 tokenizer**: Not optimal, but available offline with no downloads.
  The vocab size doesn't matter for the sparsity comparison — both models
  use the same tokenizer.
