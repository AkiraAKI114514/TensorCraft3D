"""Attention runtime shared by real training and standalone Python exports.

BSE tensors; non-causal full attention. Independent branches are averaged.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionBranch(nn.Module):
    def __init__(self, embed_dim, num_heads, kv_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
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
        result_heads = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        if observer is not None:
            observer("weighted", branch_index, {"weighted": result_heads})
        result = result_heads.transpose(1, 2).contiguous().view(batch, query_length, embed)
        branch_output = self.out_proj(result)
        if observer is not None:
            observer("output", branch_index, {"output": branch_output})
        return branch_output, projections


class TensorLabAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, kv_heads=None, dropout=0.1, attention_type="self", branches=1):
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
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.branches = nn.ModuleList([AttentionBranch(embed_dim, num_heads, kv_heads, dropout) for _ in range(branches)])

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
    def __init__(self, embed_dim, num_heads=1, kv_heads=None, ff_dim=128, dropout=0.1, attention_type="self", branches=1, norm_first=True, activation="gelu"):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("Encoder activation must be relu or gelu")
        self.norm_first = bool(norm_first)
        self.attention = TensorLabAttention(embed_dim, num_heads, kv_heads, dropout, attention_type, branches)
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
