"""Mixture-of-experts runtime shared by real training and standalone Python exports.

`TensorLabMoE` is DeepSeek-V3's MoE layer as its reference `inference/model.py` computes it on one
device (`Gate`, `Expert`, `MoE` with world_size 1): every token is scored against each routed
expert, keeps its top-k experts, and adds their SwiGLU outputs weighted by its own scores; shared
experts see every token. The layer keeps its input shape, `[..., dim]`.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

# DeepSeek-V3's auxiliary-loss-free balancing moves each expert's selection bias by this much per
# training step: up for an expert that received fewer tokens than average, down for one that
# received more. The bias only decides which experts are chosen, never how much they weigh.
BALANCE_RATE = 0.001
SCORE_FUNCTIONS = ("softmax", "sigmoid")


class TensorLabExpert(nn.Module):
    """A SwiGLU feed-forward, w2(silu(w1 x) * w3 x), with no biases: DeepSeek's `Expert` and `MLP`."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TensorLabGate(nn.Module):
    """DeepSeek-V3's `Gate`: scores, optional group-limited routing, top-k, weights.

    Scores are softmax or sigmoid over `x @ weight.T`. With `bias`, the bias is added for choosing
    only. With more than one group, only the best `topk_groups` groups stay eligible: a group is
    rated by its best score without a bias, by its two best with one. The chosen experts weigh the
    unbiased scores, renormalised to sum to 1 for sigmoid, times `route_scale`.
    """

    def __init__(self, dim, experts, top_k, score_func, route_scale, groups, topk_groups, bias):
        super().__init__()
        self.top_k, self.groups, self.topk_groups = top_k, groups, topk_groups
        self.score_func, self.route_scale = score_func, float(route_scale)
        self.weight = nn.Parameter(torch.empty(experts, dim))
        # The initialisation nn.Linear uses, since this is the same kind of projection.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.register_buffer("bias", torch.zeros(experts))
        else:
            self.bias = None

    def forward(self, x):
        scores = F.linear(x, self.weight)
        scores = scores.softmax(dim=-1, dtype=torch.float32) if self.score_func == "softmax" else scores.sigmoid()
        original = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.groups > 1:
            scores = scores.view(x.size(0), self.groups, -1)
            group_scores = scores.amax(dim=-1) if self.bias is None else scores.topk(2, dim=-1)[0].sum(dim=-1)
            chosen = group_scores.topk(self.topk_groups, dim=-1)[1]
            excluded = scores.new_ones(x.size(0), self.groups, dtype=torch.bool).scatter_(1, chosen, False)
            scores = scores.masked_fill(excluded.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.top_k, dim=-1)[1]
        weights = original.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return (weights * self.route_scale).type_as(x), indices


class TensorLabMoE(nn.Module):
    def __init__(self, dim, experts=8, top_k=2, shared_experts=1, expert_dim=64, score_func="sigmoid", route_scale=1.0, groups=1, topk_groups=1, gate_bias=True):
        super().__init__()
        counts = (dim, experts, top_k, expert_dim, groups, topk_groups)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in counts):
            raise ValueError("MoE dimensions, expert counts, top_k and groups must be positive integers")
        if isinstance(shared_experts, bool) or not isinstance(shared_experts, int) or shared_experts < 0:
            raise ValueError("shared_experts must be 0 or a positive integer")
        if top_k > experts:
            raise ValueError("top_k cannot exceed the number of routed experts")
        if experts % groups or topk_groups > groups:
            raise ValueError("experts must split evenly into groups, and topk_groups cannot exceed groups")
        if groups > 1 and top_k > topk_groups * (experts // groups):
            raise ValueError("the chosen groups must hold at least top_k experts")
        if groups > 1 and gate_bias and experts // groups < 2:
            raise ValueError("with a gate bias, each group is rated by its two best experts, so it needs at least two")
        if score_func not in SCORE_FUNCTIONS:
            raise ValueError("score_func must be softmax or sigmoid")
        if isinstance(route_scale, bool) or not isinstance(route_scale, (int, float)) or not math.isfinite(route_scale) or route_scale <= 0:
            raise ValueError("route_scale must be a positive number")
        if not isinstance(gate_bias, bool):
            raise ValueError("gate_bias must be a boolean")
        self.dim, self.experts_count = dim, experts
        self.gate = TensorLabGate(dim, experts, top_k, score_func, route_scale, groups, topk_groups, gate_bias)
        self.experts = nn.ModuleList([TensorLabExpert(dim, expert_dim) for _ in range(experts)])
        # DeepSeek folds its shared experts into one MLP that many hidden units wide.
        self.shared_experts = TensorLabExpert(dim, shared_experts * expert_dim) if shared_experts else None
        # Tokens each expert received in the last training batch; the balancing step reads it.
        self.last_load = None

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        weights, indices = self.gate(flat)
        if self.training:
            self.last_load = torch.bincount(indices.flatten(), minlength=self.experts_count).detach()
        routed = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            rows, slot = torch.where(indices == index)
            if rows.numel():
                routed = routed.index_add(0, rows, expert(flat[rows]) * weights[rows, slot, None])
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(flat)
        return routed.view(shape)

    @torch.no_grad()
    def balance(self, rate=BALANCE_RATE):
        """One step of auxiliary-loss-free balancing: nudge the gate bias towards an even load."""
        if self.gate.bias is None or self.last_load is None:
            return
        load = self.last_load.to(self.gate.bias.dtype)
        self.gate.bias += rate * torch.sign(load.mean() - load)
