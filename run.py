#!/usr/bin/env python3
"""
SGT v2 — Run: Full pipeline (train dense → train sparse → evaluate → compare).

Usage:
    python run.py --dataset /path/to/fineweb.jsonl
    python run.py --dataset /path/to/fineweb.jsonl --max_tokens 1000000000
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, description):
    print(f"\n{'━'*70}")
    print(f"  {description}")
    print(f"{'━'*70}")
    print(f"  $ {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    status = "✓" if result.returncode == 0 else "✗"
    print(f"\n  {status} {description} — {elapsed/60:.1f} min (exit {result.returncode})")
    return result.returncode == 0, elapsed


def main():
    p = argparse.ArgumentParser(description="SGT v2: Full overnight pipeline")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--max_tokens", type=int, default=500_000_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--top_k_blocks", type=int, default=16)
    p.add_argument("--ppl_samples", type=int, default=500)
    p.add_argument("--latency_tokens", type=int, default=256)
    p.add_argument("--latency_runs", type=int, default=10)
    args = p.parse_args()

    script_dir = Path(__file__).parent
    py = sys.executable

    print("=" * 70)
    print("SGT v2 — OVERNIGHT PIPELINE")
    print("=" * 70)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Tokens:     {args.max_tokens/1e6:.0f}M")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Top-k:      {args.top_k_blocks}/32 blocks ({args.top_k_blocks/32:.0%} active)")
    print(f"  Output:     {args.output_dir}")
    print(f"  Started:    {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()
    timings = {}

    ok, elapsed = run([
        py, str(script_dir / "train.py"),
        "--dataset", args.dataset,
        "--output_dir", args.output_dir,
        "--model", "dense",
        "--max_tokens", str(args.max_tokens),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
    ], "Step 1/3: Train DENSE baseline")
    timings["train_dense"] = elapsed
    if not ok:
        print("\n⚠ Dense training failed. Continuing with sparse...")

    ok, elapsed = run([
        py, str(script_dir / "train.py"),
        "--dataset", args.dataset,
        "--output_dir", args.output_dir,
        "--model", "sparse",
        "--max_tokens", str(args.max_tokens),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--top_k_blocks", str(args.top_k_blocks),
    ], "Step 2/3: Train SPARSE model")
    timings["train_sparse"] = elapsed
    if not ok:
        print("\n⚠ Sparse training failed.")

    ok, elapsed = run([
        py, str(script_dir / "eval.py"),
        "--dataset", args.dataset,
        "--output_dir", args.output_dir,
        "--dense_ckpt", str(Path(args.output_dir) / "dense" / "checkpoint.pt"),
        "--sparse_ckpt", str(Path(args.output_dir) / "sparse" / "checkpoint.pt"),
        "--ppl_samples", str(args.ppl_samples),
        "--latency_tokens", str(args.latency_tokens),
        "--latency_runs", str(args.latency_runs),
    ], "Step 3/3: Evaluate & compare")
    timings["eval"] = elapsed

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Train dense:  {timings.get('train_dense', 0)/60:>8.1f} min")
    print(f"  Train sparse: {timings.get('train_sparse', 0)/60:>8.1f} min")
    print(f"  Evaluate:     {timings.get('eval', 0)/60:>8.1f} min")
    print(f"  {'─'*35}")
    print(f"  Total:        {total/60:>8.1f} min ({total/3600:.1f} hours)")
    print(f"  Finished:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Results: {args.output_dir}/eval_results.json")


if __name__ == "__main__":
    main()
