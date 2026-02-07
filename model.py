#!/usr/bin/env python3
"""
SGT v2 — Model: Block-Sparse Transformer (150M params).

A GPT-style causal language model where each FFN layer has a lightweight
block-level router that learns which blocks of neurons to activate per token.
Unlike retrofitted sparsity, the router is trained jointly from scratch —
the model learns *how* to be sparse, not just *where* existing sparsity is.

Architecture:
    - Standard transformer decoder (RoPE, RMSNorm, SwiGLU)
    - Each FFN has a cheap block router: hidden_dim → num_blocks
    - Router produces a top-k block mask per token
    - Only selected blocks compute gate/up/down projections
    - Straight-through estimator for gradient flow through discrete mask

Config (150M params):
    - 12 layers, hidden_dim=768, intermediate_dim=2048
    - 12 attention heads, head_dim=64
    - block_size=64, num_blocks=32, top_k=16 (50% target sparsity)
    - Vocab=32000, max_seq_len=1024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Configuration
# ============================================================

@dataclass
class SGTConfig:
    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 768
    intermediate_dim: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    max_seq_len: int = 1024

    # Block-sparse FFN
    block_size: int = 64
    num_blocks: int = 32           # intermediate_dim // block_size
    top_k_blocks: int = 16         # 50% sparsity target
    router_dim: int = 128          # hidden → router_dim → num_blocks

    # Training
    dropout: float = 0.0
    rope_theta: float = 10000.0

    # Auxiliary loss weight for load balancing
    aux_loss_weight: float = 0.01

    def __post_init__(self):
        assert self.intermediate_dim == self.num_blocks * self.block_size, \
            f"intermediate_dim ({self.intermediate_dim}) must equal num_blocks ({self.num_blocks}) × block_size ({self.block_size})"
        assert self.hidden_dim == self.num_heads * self.head_dim, \
            f"hidden_dim ({self.hidden_dim}) must equal num_heads ({self.num_heads}) × head_dim ({self.head_dim})"


# ============================================================
# Components
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ============================================================
# Attention
# ============================================================

class Attention(nn.Module):
    def __init__(self, config: SGTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x, attention_mask=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use PyTorch SDPA for efficiency
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=True,
            dropout_p=0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(attn_out)


# ============================================================
# Block-Sparse FFN — the core innovation
# ============================================================

class BlockSparseFFN(nn.Module):
    """SwiGLU FFN with a trainable block-level router.

    Instead of computing all intermediate neurons, a small router network
    selects the top-k blocks per token. Only those blocks participate in
    the gate/up/down computation.

    The router is trained end-to-end with straight-through estimation:
    forward pass uses hard top-k mask, backward pass uses soft sigmoid
    gradients. This lets the model learn which blocks to activate.

    Block structure:
        gate_proj.weight: [intermediate_dim, hidden_dim] — stored as [num_blocks, block_size, hidden_dim]
        up_proj.weight:   [intermediate_dim, hidden_dim] — same
        down_proj.weight: [hidden_dim, intermediate_dim] — stored as [hidden_dim, num_blocks, block_size]
    """

    def __init__(self, config: SGTConfig):
        super().__init__()
        self.config = config
        self.num_blocks = config.num_blocks
        self.block_size = config.block_size
        self.top_k = config.top_k_blocks
        self.intermediate_dim = config.intermediate_dim

        # Standard SwiGLU weights
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

        # Lightweight router: hidden → router_dim → num_blocks
        self.router = nn.Sequential(
            nn.Linear(config.hidden_dim, config.router_dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.router_dim, config.num_blocks, bias=False),
        )

        # Track block selection frequency for load balancing
        self.register_buffer("block_counts", torch.zeros(config.num_blocks))
        self.register_buffer("total_tokens", torch.zeros(1))

    def forward(self, x):
        B, S, D = x.shape
        N = B * S
        x_flat = x.reshape(N, D)

        # ── Route ──
        router_logits = self.router(x_flat)               # [N, num_blocks]
        router_probs = torch.sigmoid(router_logits)         # soft probabilities

        # Hard top-k selection
        _, top_indices = router_logits.topk(self.top_k, dim=-1)  # [N, top_k]
        hard_mask = torch.zeros_like(router_logits)
        hard_mask.scatter_(1, top_indices, 1.0)              # [N, num_blocks] binary

        # Straight-through: forward uses hard mask, backward uses soft probs
        mask = hard_mask - router_probs.detach() + router_probs  # [N, num_blocks]

        # Track load balance (no grad)
        if self.training:
            with torch.no_grad():
                self.block_counts += hard_mask.sum(0)
                self.total_tokens += N

        # ── Sparse SwiGLU ──
        # Full computation (we mask after — on GPU this is more efficient than gather
        # at our scale; the mask prevents gradient flow to inactive blocks)
        gate_out = F.silu(self.gate_proj(x_flat))           # [N, I]
        up_out = self.up_proj(x_flat)                        # [N, I]

        # Apply block mask: expand [N, num_blocks] → [N, intermediate_dim]
        block_mask = mask.repeat_interleave(self.block_size, dim=1)  # [N, I]
        hidden = gate_out * up_out * block_mask              # masked SwiGLU

        output = self.down_proj(hidden)                      # [N, D]
        return output.view(B, S, D)

    def aux_loss(self):
        """Load balancing loss — encourages uniform block utilization.

        Without this, the router collapses to always selecting the same blocks,
        wasting capacity. We penalise deviation from uniform selection.
        """
        if self.total_tokens.item() == 0:
            return torch.tensor(0.0, device=self.block_counts.device)

        freq = self.block_counts / self.total_tokens          # [num_blocks]
        target = self.top_k / self.num_blocks                  # expected uniform rate
        loss = ((freq - target) ** 2).mean()
        return loss

    def reset_stats(self):
        self.block_counts.zero_()
        self.total_tokens.zero_()

    def get_effective_sparsity(self):
        """What fraction of blocks are inactive on average."""
        return 1.0 - self.top_k / self.num_blocks


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, config: SGTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = BlockSparseFFN(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ============================================================
# Full Model
# ============================================================

class SGTModel(nn.Module):
    """Block-Sparse GPT — a causal LM with trainable block-sparse FFNs."""

    def __init__(self, config: SGTConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        # Scale residual projections
        for layer in self.layers:
            layer.attn.o_proj.weight.data *= (2 * config.num_layers) ** -0.5
            layer.ffn.down_proj.weight.data *= (2 * config.num_layers) ** -0.5

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Add auxiliary load-balancing loss
            aux = sum(layer.ffn.aux_loss() for layer in self.layers) / self.config.num_layers
            loss = loss + self.config.aux_loss_weight * aux

        return {"logits": logits, "loss": loss}

    def reset_router_stats(self):
        for layer in self.layers:
            layer.ffn.reset_stats()

    def get_sparsity_stats(self):
        """Return per-layer block utilization and effective sparsity."""
        stats = {}
        for i, layer in enumerate(self.layers):
            ffn = layer.ffn
            if ffn.total_tokens.item() > 0:
                freq = ffn.block_counts / ffn.total_tokens
                stats[i] = {
                    "mean_activation_rate": freq.mean().item(),
                    "std_activation_rate": freq.std().item(),
                    "min_block_freq": freq.min().item(),
                    "max_block_freq": freq.max().item(),
                    "target_sparsity": ffn.get_effective_sparsity(),
                }
        return stats

    def num_parameters(self, exclude_embeddings=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n


# ============================================================
# Dense baseline for comparison
# ============================================================

class DenseFFN(nn.Module):
    """Standard SwiGLU FFN — no routing, no sparsity."""

    def __init__(self, config: SGTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def aux_loss(self):
        return torch.tensor(0.0)

    def reset_stats(self):
        pass

    def get_effective_sparsity(self):
        return 0.0


class DenseTransformerBlock(nn.Module):
    def __init__(self, config: SGTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = DenseFFN(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DenseModel(nn.Module):
    """Dense baseline — identical architecture but without block routing."""

    def __init__(self, config: SGTConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([DenseTransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        for layer in self.layers:
            layer.attn.o_proj.weight.data *= (2 * config.num_layers) ** -0.5
            layer.ffn.down_proj.weight.data *= (2 * config.num_layers) ** -0.5

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    def reset_router_stats(self):
        pass

    def get_sparsity_stats(self):
        return {}

    def num_parameters(self, exclude_embeddings=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n
