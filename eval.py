#!/usr/bin/env python3
"""
SGT v2 — Eval: Compare dense and sparse checkpoints.

Measures perplexity, decode latency, and block utilisation patterns.

Usage:
    python eval.py --dataset /path/to/fineweb.jsonl
    python eval.py --dataset /path/to/fineweb.jsonl --dense_ckpt outputs/dense/checkpoint.pt --sparse_ckpt outputs/sparse/checkpoint.pt
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import SGTConfig, SGTModel, DenseModel


# ============================================================
# Tokenizer
# ============================================================

def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


# ============================================================
# Load checkpoint
# ============================================================

def load_model(ckpt_path, model_class):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = SGTConfig(**ckpt["config"])
    model = model_class(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda().eval()
    return model, config


# ============================================================
# Perplexity evaluation
# ============================================================

@torch.no_grad()
def eval_ppl(model, dataset_path, tokenizer, config, num_samples=500, model_type="dense"):
    """Evaluate perplexity on held-out data."""
    total_loss = 0
    total_tokens = 0
    seq_len = config.max_seq_len

    buffer = []
    samples_evaluated = 0

    # Skip first 100K lines (training data), use next samples for eval
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines_skipped = 0
        for line in f:
            lines_skipped += 1
            if lines_skipped < 100000:
                continue

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "")
            if len(text) < 50:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= seq_len + 1:
                chunk = buffer[:seq_len + 1]
                buffer = buffer[seq_len:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0).cuda()
                labels = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0).cuda()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)

                loss = outputs["loss"]
                n_tok = seq_len
                total_loss += loss.item() * n_tok
                total_tokens += n_tok
                samples_evaluated += 1

                if samples_evaluated >= num_samples:
                    break

            if samples_evaluated >= num_samples:
                break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl, total_tokens


# ============================================================
# Latency benchmark
# ============================================================

@torch.no_grad()
def benchmark_latency(model, config, tokenizer, num_tokens=256, warmup=5, runs=10):
    """Measure decode latency at batch size 1."""
    prompt = "The fundamental principle of block-sparse computation is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_ids = input_ids.cuda()

    # Warmup
    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                outputs = model(input_ids=input_ids)
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids_tmp = torch.cat([input_ids, next_token], dim=1)

    # Timed runs
    latencies = []
    for _ in range(runs):
        ids = input_ids.clone()
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(num_tokens):
                outputs = model(input_ids=ids)
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_token], dim=1)
                if ids.shape[1] > config.max_seq_len:
                    ids = ids[:, -config.max_seq_len:]

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        latencies.append(elapsed)

    latencies = np.array(latencies)
    tok_per_sec = num_tokens / latencies
    ms_per_tok = latencies / num_tokens * 1000

    return {
        "num_tokens": num_tokens,
        "runs": runs,
        "tok_per_sec_mean": float(tok_per_sec.mean()),
        "tok_per_sec_std": float(tok_per_sec.std()),
        "ms_per_tok_mean": float(ms_per_tok.mean()),
        "ms_per_tok_std": float(ms_per_tok.std()),
    }


# ============================================================
# Block utilisation analysis (sparse model only)
# ============================================================

@torch.no_grad()
def analyse_block_utilisation(model, dataset_path, tokenizer, config, num_samples=200):
    """Run data through sparse model and collect block selection patterns."""
    model.reset_router_stats()

    buffer = []
    samples = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines_skipped = 0
        for line in f:
            lines_skipped += 1
            if lines_skipped < 100000:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if len(text) < 50:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= config.max_seq_len:
                chunk = buffer[:config.max_seq_len]
                buffer = buffer[config.max_seq_len:]
                input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).cuda()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=input_ids)

                samples += 1
                if samples >= num_samples:
                    break
            if samples >= num_samples:
                break

    stats = model.get_sparsity_stats()
    return stats


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description="SGT v2: Evaluate dense vs sparse")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dense_ckpt", type=str, default="outputs/dense/checkpoint.pt")
    p.add_argument("--sparse_ckpt", type=str, default="outputs/sparse/checkpoint.pt")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--ppl_samples", type=int, default=500)
    p.add_argument("--latency_tokens", type=int, default=256)
    p.add_argument("--latency_runs", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    tokenizer = get_tokenizer()
    results = {}

    # ── Dense ──
    if Path(args.dense_ckpt).exists():
        print(f"\n{'='*70}")
        print("EVALUATING: DENSE")
        print(f"{'='*70}")
        model, config = load_model(args.dense_ckpt, DenseModel)
        n_params = model.num_parameters()
        print(f"  Parameters: {n_params:,}")

        print("\n  Perplexity...")
        loss, ppl, tokens = eval_ppl(model, args.dataset, tokenizer, config, args.ppl_samples)
        print(f"  Loss: {loss:.4f}, PPL: {ppl:.4f} ({tokens:,} tokens)")

        print("\n  Latency benchmark...")
        latency = benchmark_latency(model, config, tokenizer, args.latency_tokens, runs=args.latency_runs)
        print(f"  {latency['tok_per_sec_mean']:.1f} +/- {latency['tok_per_sec_std']:.1f} tok/s")
        print(f"  {latency['ms_per_tok_mean']:.2f} +/- {latency['ms_per_tok_std']:.2f} ms/tok")

        results["dense"] = {
            "params": n_params,
            "loss": loss, "ppl": ppl,
            "latency": latency,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"  Warning: Dense checkpoint not found: {args.dense_ckpt}")

    # ── Sparse ──
    if Path(args.sparse_ckpt).exists():
        print(f"\n{'='*70}")
        print("EVALUATING: SPARSE")
        print(f"{'='*70}")
        model, config = load_model(args.sparse_ckpt, SGTModel)
        n_params = model.num_parameters()
        router_params = sum(
            sum(p.numel() for p in layer.ffn.router.parameters())
            for layer in model.layers
        )
        print(f"  Parameters: {n_params:,} (router: {router_params:,})")
        print(f"  Target sparsity: {1 - config.top_k_blocks/config.num_blocks:.0%}")

        print("\n  Perplexity...")
        loss, ppl, tokens = eval_ppl(model, args.dataset, tokenizer, config, args.ppl_samples, "sparse")
        print(f"  Loss: {loss:.4f}, PPL: {ppl:.4f} ({tokens:,} tokens)")

        print("\n  Block utilisation analysis...")
        block_stats = analyse_block_utilisation(model, args.dataset, tokenizer, config)
        print(f"\n  Per-layer block utilisation:")
        for layer_idx, stats in sorted(block_stats.items()):
            print(f"    Layer {layer_idx:>2}: mean_rate={stats['mean_activation_rate']:.3f} "
                  f"std={stats['std_activation_rate']:.4f} "
                  f"min={stats['min_block_freq']:.3f} max={stats['max_block_freq']:.3f}")

        print("\n  Latency benchmark...")
        latency = benchmark_latency(model, config, tokenizer, args.latency_tokens, runs=args.latency_runs)
        print(f"  {latency['tok_per_sec_mean']:.1f} +/- {latency['tok_per_sec_std']:.1f} tok/s")
        print(f"  {latency['ms_per_tok_mean']:.2f} +/- {latency['ms_per_tok_std']:.2f} ms/tok")

        results["sparse"] = {
            "params": n_params,
            "router_params": router_params,
            "loss": loss, "ppl": ppl,
            "latency": latency,
            "block_utilisation": {str(k): v for k, v in block_stats.items()},
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"  Warning: Sparse checkpoint not found: {args.sparse_ckpt}")

    # ── Comparison ──
    if "dense" in results and "sparse" in results:
        d = results["dense"]
        s = results["sparse"]
        ppl_gap = (s["ppl"] - d["ppl"]) / d["ppl"] * 100
        speedup = s["latency"]["tok_per_sec_mean"] / d["latency"]["tok_per_sec_mean"]

        print(f"\n{'='*70}")
        print("COMPARISON: DENSE vs SPARSE")
        print(f"{'='*70}")
        print(f"\n  {'Metric':<25} {'Dense':>15} {'Sparse':>15} {'Delta':>15}")
        print(f"  {'-'*70}")
        print(f"  {'PPL':<25} {d['ppl']:>15.4f} {s['ppl']:>15.4f} {ppl_gap:>+14.2f}%")
        print(f"  {'Throughput (tok/s)':<25} {d['latency']['tok_per_sec_mean']:>15.1f} {s['latency']['tok_per_sec_mean']:>15.1f} {speedup:>14.2f}x")
        print(f"  {'Latency (ms/tok)':<25} {d['latency']['ms_per_tok_mean']:>15.2f} {s['latency']['ms_per_tok_mean']:>15.2f}")
        print(f"  {'Parameters':<25} {d['params']:>15,} {s['params']:>15,}")

        print(f"\n  Verdict:")
        if abs(ppl_gap) <= 2.0 and speedup > 1.0:
            print(f"  SUCCESS: Sparse matches dense PPL (gap {ppl_gap:+.2f}%) "
                  f"with {speedup:.2f}x throughput")
        elif abs(ppl_gap) <= 5.0:
            print(f"  PARTIAL: PPL gap {ppl_gap:+.2f}%, speedup {speedup:.2f}x")
        else:
            print(f"  PPL gap too large: {ppl_gap:+.2f}%")

    # Save
    eval_path = out_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {eval_path}")


if __name__ == "__main__":
    main()
