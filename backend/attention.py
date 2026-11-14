"""Attention runtime shared by real training and standalone Python exports.

BSE tensors. Attention is full by default; `causal=True` masks every key position after
the query's own, for self-attention only. Independent branches are averaged.

`TensorLabLatentAttention` is DeepSeek-V2/V3's Multi-head Latent Attention (MLA), a separate
operator: Q and K/V pass through low-rank latents, and a decoupled rotary part carries position.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class AttentionBranch(nn.Module):
    def __init__(self, embed_dim, num_heads, kv_heads, dropout, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = bool(causal)
        kv_dim = self.head_dim * kv_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, kv_dim)
        self.v_proj = nn.Linear(embed_dim, kv_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context, overrides=None, prefix="", observer=None, branch_index=0):
        batch, query_length, embed = query.shape
        overrides = overrides or {}
        projections = {}
        def project(role, layer, source, count):
            default = layer(source).view(batch, source.shape[1], count, self.head_dim)
            heads = []
            for index in range(count):
                port = f"{prefix}{role}{index}"
                if port in overrides:
                    start = index * self.head_dim
                    tensor = F.linear(overrides[port], layer.weight[start:start + self.head_dim], layer.bias[start:start + self.head_dim])
                else:
                    tensor = default[:, :, index]
                projections[port] = tensor
                heads.append(tensor)
            return torch.stack(heads, dim=1)
        q = project("q", self.q_proj, query, self.num_heads)
        k_native = project("k", self.k_proj, context, self.kv_heads)
        v_native = project("v", self.v_proj, context, self.kv_heads)
        if observer is not None:
            observer("q", branch_index, {"q": q})
            observer("k", branch_index, {"k": k_native})
            observer("v", branch_index, {"v": v_native})
        # Each KV group serves num_heads / kv_heads consecutive Q heads.
        repeats = self.num_heads // self.kv_heads
        k = k_native.repeat_interleave(repeats, dim=1) if repeats > 1 else k_native
        v = v_native.repeat_interleave(repeats, dim=1) if repeats > 1 else v_native
        # A causal mask is square by definition: position t may attend to keys 0..t. A K/V
        # override of another length would make torch build a rectangular lower triangle that
        # means nothing, so refuse it here as well as in the graph analyzers.
        if self.causal and k.shape[-2] != query_length:
            raise ValueError("Causal attention requires the key length to equal the query length")
        result_heads = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        if observer is not None:
            observer("weighted", branch_index, {"weighted": result_heads})
        result = result_heads.transpose(1, 2).contiguous().view(batch, query_length, embed)
        branch_output = self.out_proj(result)
        if observer is not None:
            observer("output", branch_index, {"output": branch_output})
        return branch_output, projections


class TensorLabAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, kv_heads=None, dropout=0.1, attention_type="self", branches=1, causal=False):
        super().__init__()
        kv_heads = (1 if attention_type == "multi_query" else num_heads) if kv_heads is None else kv_heads
        if attention_type not in ("self", "multi_query", "grouped_query", "cross", "multi_branch"):
            raise ValueError("Unknown attention_type")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (embed_dim, num_heads, kv_heads, branches)):
            raise ValueError("Attention dimensions, heads and branches must be positive integers")
        if embed_dim % num_heads or kv_heads > num_heads or num_heads % kv_heads:
            raise ValueError("embed_dim / num_heads and num_heads / kv_heads must be integers")
        if attention_type in ("self", "multi_branch") and kv_heads != num_heads:
            raise ValueError("Self attention requires kv_heads = num_heads")
        if attention_type == "multi_query" and kv_heads != 1:
            raise ValueError("MQA requires kv_heads = 1")
        if attention_type == "multi_branch" and branches < 2:
            raise ValueError("Multi-Branch attention requires at least two branches")
        if not isinstance(causal, bool):
            raise ValueError("causal must be a boolean")
        # Query and Context are different sequences in cross-attention, so "positions after
        # the query's own" has no meaning there.
        if causal and attention_type == "cross":
            raise ValueError("Causal masking applies to self-attention only")
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.causal = causal
        self.branches = nn.ModuleList([AttentionBranch(embed_dim, num_heads, kv_heads, dropout, causal) for _ in range(branches)])

    def forward(self, query, context=None, observer=None):
        return self.forward_with_ports(query, context, observer=observer)[0]

    def forward_with_ports(self, query, context=None, overrides=None, observer=None):
        if self.attention_type == "cross":
            if context is None:
                raise ValueError("Cross-Attention requires Query and Context inputs")
        else:
            if context is not None:
                raise ValueError("Self attention takes a single input")
            context = query
        result = None
        projections = {}
        for index, branch in enumerate(self.branches):
            value, ports = branch(query, context, overrides, f"b{index}:", observer, index)
            result = value if result is None else result + value
            projections.update(ports)
        merged = result / len(self.branches)
        if observer is not None:
            observer("merged", -1, {"output": merged})
        return merged, projections


class TensorLabTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=1, kv_heads=None, ff_dim=128, dropout=0.1, attention_type="self", branches=1, norm_first=True, activation="gelu", causal=False):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("Encoder activation must be relu or gelu")
        self.norm_first = bool(norm_first)
        self.attention = TensorLabAttention(embed_dim, num_heads, kv_heads, dropout, attention_type, branches, causal)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU() if activation == "relu" else nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, query, context=None, observer=None):
        return self.forward_with_ports(query, context, observer=observer)[0]

    def forward_with_ports(self, query, context=None, overrides=None, observer=None):
        # Q and self-attention overrides receive the same pre-norm as the base input.
        overrides = {port: self.norm1(value) if self.norm_first and (":q" in port or self.attention.attention_type != "cross") else value for port, value in (overrides or {}).items()}
        attention, ports = self.attention.forward_with_ports(self.norm1(query) if self.norm_first else query, context, overrides, observer=observer)
        x = query + self.dropout(attention)
        if self.norm_first:
            return x + self.ffn(self.norm2(x)), ports
        x = self.norm1(x)
        return self.norm2(x + self.ffn(x)), ports


def rotary_pairs(x, theta):
    """Rotate adjacent pairs (x[2i], x[2i+1]) of the last axis by position, as DeepSeek-V3's
    `apply_rotary_emb` does with complex numbers: pair i at position t turns by
    t / theta^(2i/d). `x` is [B, S, H, d] with an even d; the result has x's shape and dtype."""
    length, width = x.shape[1], x.shape[-1]
    inverse = 1.0 / (theta ** (torch.arange(0, width, 2, dtype=torch.float32, device=x.device) / width))
    angles = torch.outer(torch.arange(length, dtype=torch.float32, device=x.device), inverse)
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    pairs = x.float().reshape(*x.shape[:-1], width // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    # (a + ib)(cos + i sin) = (a cos - b sin) + i(a sin + b cos), interleaved back in place.
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2).to(x.dtype)


class TensorLabLatentAttention(nn.Module):
    """Multi-head Latent Attention as DeepSeek-V3's `inference/model.py` computes it without a KV
    cache (its "naive" path) and without YaRN scaling.

    - Q comes from a low-rank latent (`wq_a` -> RMSNorm -> `wq_b`), or from one projection when
      `q_lora_rank` is 0.
    - K/V share a second latent (`wkv_a` -> RMSNorm -> `wkv_b`). `wkv_a` also emits one rotary key
      part that every head shares.
    - Each head's query and key are a position-free part and a rotary part, concatenated, so the
      score is q.k * (qk_nope_head_dim + qk_rope_head_dim)^-0.5.
    - None of the projections has a bias. Self-attention only; `causal` masks later positions.
    """

    def __init__(self, embed_dim, num_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, rope_theta=10000.0, causal=True, eps=1e-6):
        super().__init__()
        sizes = (embed_dim, num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in sizes):
            raise ValueError("Latent attention dimensions and heads must be positive integers")
        if isinstance(q_lora_rank, bool) or not isinstance(q_lora_rank, int) or q_lora_rank < 0:
            raise ValueError("q_lora_rank must be 0 (no query latent) or a positive integer")
        if qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even: the rotary part turns pairs of values")
        if isinstance(rope_theta, bool) or not isinstance(rope_theta, (int, float)) or not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be a positive number")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive number")
        if not isinstance(causal, bool):
            raise ValueError("causal must be a boolean")
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = float(rope_theta)
        self.causal = causal
        if q_lora_rank:
            self.wq_a = nn.Linear(embed_dim, q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(q_lora_rank, eps=eps)
            self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        else:
            self.wq = nn.Linear(embed_dim, num_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(embed_dim, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank, eps=eps)
        self.wkv_b = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)
        self.wo = nn.Linear(num_heads * v_head_dim, embed_dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, x, observer=None):
        batch, length, _ = x.shape
        heads = self.num_heads
        q = self.wq_b(self.q_norm(self.wq_a(x))) if self.q_lora_rank else self.wq(x)
        q_nope, q_pe = torch.split(q.view(batch, length, heads, self.qk_head_dim), [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent, k_pe = torch.split(self.wkv_a(x), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = rotary_pairs(k_pe.unsqueeze(2), self.rope_theta)
        kv = self.wkv_b(self.kv_norm(latent)).view(batch, length, heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # [B, H, S, d], the layout the shared attention observation reads.
        q = torch.cat([q_nope, rotary_pairs(q_pe, self.rope_theta)], dim=-1).transpose(1, 2)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, heads, -1)], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        if observer is not None:
            observer("q", 0, {"q": q})
            observer("k", 0, {"k": k})
            observer("v", 0, {"v": v})
        heads_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=self.softmax_scale)
        if observer is not None:
            observer("weighted", 0, {"weighted": heads_out})
        output = self.wo(heads_out.transpose(1, 2).reshape(batch, length, heads * self.v_head_dim))
        if observer is not None:
            observer("output", 0, {"output": output})
            observer("merged", -1, {"output": output})
        return output
