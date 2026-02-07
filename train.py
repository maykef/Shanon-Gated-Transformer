#!/usr/bin/env python3
"""
SGT v2 — Train: Train dense baseline and block-sparse model side by side.

Trains both models on the same data with identical hyperparameters so the
only variable is the block-sparse router. Logs loss curves, sparsity stats,
and router utilisation throughout training.

Usage:
    python train.py --dataset /path/to/fineweb.jsonl
    python train.py --dataset /path/to/fineweb.jsonl --model sparse --epochs 1
    python train.py --dataset /path/to/fineweb.jsonl --model both

Outputs:
    outputs/{dense,sparse}/
        checkpoint.pt       — model weights
        train_log.json      — loss curves and sparsity stats
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from model import SGTConfig, SGTModel, DenseModel


# ============================================================
# Streaming dataset — no need to load everything into memory
# ============================================================

class FineWebStreamDataset(IterableDataset):
    """Streams tokenised chunks from a JSONL file.

    Packs multiple documents into fixed-length sequences for efficiency.
    No padding, no wasted tokens.
    """

    def __init__(self, path, tokenizer, seq_len=1024, max_tokens=None):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer = []
        total_tokens = 0

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
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

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(tokens)

                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": labels}

                    total_tokens += self.seq_len
                    if self.max_tokens and total_tokens >= self.max_tokens:
                        return


# ============================================================
# Tokenizer — use a standard one for 32k vocab
# ============================================================

def get_tokenizer():
    """Get a 32k vocab tokenizer. Use GPT-2's as a simple default."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # GPT-2 has 50257 vocab — we'll adjust config to match
    return tokenizer


# ============================================================
# Training loop
# ============================================================

def train_model(model_type, config, args):
    out_dir = Path(args.output_dir) / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TRAINING: {model_type.upper()}")
    print(f"{'='*70}")

    # ── Build model ──
    if model_type == "sparse":
        model = SGTModel(config).cuda()
    else:
        model = DenseModel(config).cuda()

    n_params = model.num_parameters()
    n_params_noembeddings = model.num_parameters(exclude_embeddings=True)
    print(f"  Parameters: {n_params:,} total, {n_params_noembeddings:,} non-embedding")

    if model_type == "sparse":
        router_params = sum(
            sum(p.numel() for p in layer.ffn.router.parameters())
            for layer in model.layers
        )
        print(f"  Router parameters: {router_params:,} ({router_params/n_params*100:.2f}%)")
        print(f"  Target sparsity: {1 - config.top_k_blocks/config.num_blocks:.0%}")

    # ── Data ──
    tokenizer = get_tokenizer()
    dataset = FineWebStreamDataset(
        args.dataset, tokenizer, seq_len=config.max_seq_len,
        max_tokens=args.max_tokens,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2,
                        prefetch_factor=4, pin_memory=True)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.1, fused=True,
    )

    # Cosine decay with warmup
    total_steps = args.max_steps
    warmup_steps = min(500, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # decay to 10% of peak

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Compile for speed ──
    if hasattr(torch, "compile") and args.compile:
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Training ──
    model.train()
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    log = {
        "model_type": model_type,
        "config": {k: v for k, v in config.__dict__.items()},
        "args": vars(args),
        "steps": [],
    }

    global_step = 0
    total_loss = 0
    tokens_processed = 0
    t0 = time.time()
    log_interval = args.log_interval

    pbar = tqdm(total=total_steps, desc=f"{model_type}")

    for batch in loader:
        if global_step >= total_steps:
            break

        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item()
        tokens_processed += input_ids.numel()
        global_step += 1

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
            tok=f"{tokens_processed/1e6:.1f}M",
        )

        if global_step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed

            entry = {
                "step": global_step,
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
                "tokens": tokens_processed,
                "tok_per_sec": tok_per_sec,
                "elapsed_min": elapsed / 60,
            }

            # Sparsity stats for sparse model
            if model_type == "sparse":
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                stats = raw_model.get_sparsity_stats()
                if stats:
                    mean_rates = [s["mean_activation_rate"] for s in stats.values()]
                    std_rates = [s["std_activation_rate"] for s in stats.values()]
                    entry["mean_block_activation_rate"] = float(np.mean(mean_rates))
                    entry["block_activation_std"] = float(np.mean(std_rates))
                    entry["effective_sparsity"] = 1.0 - entry["mean_block_activation_rate"]
                raw_model.reset_router_stats()

            log["steps"].append(entry)
            total_loss = 0

            if global_step % (log_interval * 10) == 0:
                ppl = math.exp(min(avg_loss, 20))
                print(f"\n  Step {global_step}: loss={avg_loss:.4f} ppl={ppl:.2f} "
                      f"lr={scheduler.get_last_lr()[0]:.1e} "
                      f"tok/s={tok_per_sec:.0f} "
                      f"tokens={tokens_processed/1e6:.1f}M")
                if model_type == "sparse" and "effective_sparsity" in entry:
                    print(f"    Effective sparsity: {entry['effective_sparsity']:.1%} "
                          f"(activation std: {entry['block_activation_std']:.4f})")

    pbar.close()
    elapsed = time.time() - t0

    # ── Final stats ──
    final_loss = log["steps"][-1]["loss"] if log["steps"] else 0
    final_ppl = math.exp(min(final_loss, 20))

    print(f"\n  ✓ Training complete: {elapsed/60:.1f} min, "
          f"{tokens_processed:,} tokens, "
          f"final loss={final_loss:.4f}, ppl={final_ppl:.2f}")

    log["final"] = {
        "loss": final_loss,
        "ppl": final_ppl,
        "tokens_processed": tokens_processed,
        "elapsed_min": elapsed / 60,
        "tok_per_sec": tokens_processed / elapsed,
    }

    # ── Save ──
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "config": config.__dict__,
        "step": global_step,
    }, out_dir / "checkpoint.pt")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"  ✓ Saved: {out_dir}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return log


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description="SGT v2: Train block-sparse transformer")
    p.add_argument("--dataset", type=str, required=True, help="FineWeb JSONL path")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--model", type=str, default="both", choices=["dense", "sparse", "both"])
    p.add_argument("--max_tokens", type=int, default=500_000_000, help="Total training tokens")
    p.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides max_tokens)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--log_interval", type=int, default=50)

    # Architecture overrides
    p.add_argument("--top_k_blocks", type=int, default=16, help="Blocks to activate (sparsity control)")
    p.add_argument("--aux_loss_weight", type=float, default=0.01)

    args = p.parse_args()

    # Config
    tokenizer = get_tokenizer()
    config = SGTConfig(
        vocab_size=tokenizer.vocab_size,
        top_k_blocks=args.top_k_blocks,
        aux_loss_weight=args.aux_loss_weight,
    )

    if args.max_steps is None:
        args.max_steps = args.max_tokens // (args.batch_size * config.max_seq_len)

    print("=" * 70)
    print("SGT v2 — BLOCK-SPARSE TRANSFORMER TRAINING")
    print("=" * 70)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Vocab:       {config.vocab_size}")
    print(f"  Model:       {config.num_layers}L / {config.hidden_dim}H / {config.intermediate_dim}I")
    print(f"  Blocks:      {config.num_blocks} × {config.block_size}, top-k={config.top_k_blocks}")
    print(f"  Batch:       {args.batch_size} × {config.max_seq_len} = {args.batch_size * config.max_seq_len:,} tokens/step")
    print(f"  Steps:       {args.max_steps:,}")
    print(f"  Tokens:      ~{args.max_steps * args.batch_size * config.max_seq_len / 1e6:.0f}M")
    print(f"  LR:          {args.lr}")
    print(f"  Compile:     {args.compile}")
    print(f"  AMP:         {args.amp}")

    all_logs = {}

    if args.model in ("dense", "both"):
        all_logs["dense"] = train_model("dense", config, args)

    if args.model in ("sparse", "both"):
        all_logs["sparse"] = train_model("sparse", config, args)

    # ── Comparison ──
    if len(all_logs) == 2:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        d = all_logs["dense"]["final"]
        s = all_logs["sparse"]["final"]
        print(f"\n  {'Metric':<25} {'Dense':>15} {'Sparse':>15}")
        print(f"  {'-'*55}")
        print(f"  {'Final loss':<25} {d['loss']:>15.4f} {s['loss']:>15.4f}")
        print(f"  {'Final PPL':<25} {d['ppl']:>15.2f} {s['ppl']:>15.2f}")
        print(f"  {'Training time (min)':<25} {d['elapsed_min']:>15.1f} {s['elapsed_min']:>15.1f}")
        print(f"  {'Throughput (tok/s)':<25} {d['tok_per_sec']:>15.0f} {s['tok_per_sec']:>15.0f}")
        print(f"  {'Tokens processed':<25} {d['tokens_processed']:>15,} {s['tokens_processed']:>15,}")

        ppl_gap = (s["ppl"] - d["ppl"]) / d["ppl"] * 100
        print(f"\n  PPL gap (sparse vs dense): {ppl_gap:+.2f}%")
        if abs(ppl_gap) < 2:
            print(f"  ✓ Within 2% — sparse model matches dense quality")
        elif ppl_gap > 0:
            print(f"  ⚠ Sparse model is {ppl_gap:.1f}% worse in PPL")
        else:
            print(f"  ✓ Sparse model is {-ppl_gap:.1f}% better in PPL")

    # Save combined summary
    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: v.get("final", {}) for k, v in all_logs.items()}, f, indent=2)
    print(f"\n✓ Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
