"""Mixture of experts (MoE): parity with DeepSeek-V3's reference routing, analysis, balancing,
training and source import.

The tests are skipped when the optional PyTorch training extra is unavailable.
"""
import threading
import unittest

from .graph import analyze_graph, moe_parameters
from .pytorch_import import import_pytorch
from .training import build_model, train

try:
    import torch
    from torch.nn import functional as F

    from .moe import BALANCE_RATE, TensorLabMoE
except ImportError:  # pragma: no cover - deployment without training extra
    torch = None


def deepseek_gate(gate, x):
    """DeepSeek-V3 `inference/model.py` `Gate.forward`, reading `gate`'s weight and bias."""
    scores = F.linear(x, gate.weight)
    if gate.score_func == "softmax":
        scores = scores.softmax(dim=-1, dtype=torch.float32)
    else:
        scores = scores.sigmoid()
    original_scores = scores
    if gate.bias is not None:
        scores = scores + gate.bias
    if gate.groups > 1:
        scores = scores.view(x.size(0), gate.groups, -1)
        if gate.bias is None:
            group_scores = scores.amax(dim=-1)
        else:
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
        indices = group_scores.topk(gate.topk_groups, dim=-1)[1]
        mask = scores.new_ones(x.size(0), gate.groups, dtype=bool).scatter_(1, indices, False)
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
    indices = torch.topk(scores, gate.top_k, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if gate.score_func == "sigmoid":
        weights /= weights.sum(dim=-1, keepdim=True)
    weights *= gate.route_scale
    return weights.type_as(x), indices


def deepseek_expert(expert, x):
    """DeepSeek-V3 `Expert.forward` / `MLP.forward`: w2(silu(w1(x)) * w3(x))."""
    return F.linear(F.silu(F.linear(x, expert.w1.weight)) * F.linear(x, expert.w3.weight), expert.w2.weight)


def deepseek_moe(module, x):
    """DeepSeek-V3 `MoE.forward` with world_size 1, reading `module`'s weights."""
    shape = x.size()
    x = x.view(-1, module.dim)
    weights, indices = deepseek_gate(module.gate, x)
    y = torch.zeros_like(x)
    counts = torch.bincount(indices.flatten(), minlength=len(module.experts)).tolist()
    for i in range(len(module.experts)):
        if counts[i] == 0:
            continue
        idx, top = torch.where(indices == i)
        y[idx] += deepseek_expert(module.experts[i], x[idx]) * weights[idx, top, None]
    z = deepseek_expert(module.shared_experts, x) if module.shared_experts is not None else 0
    return (y + z).view(shape)


def moe_graph(params=None):
    layer = {"experts": 4, "top_k": 2, "shared_experts": 1, "expert_dim": 8, **(params or {})}
    return {"version": 1, "name": "MoE", "nodes": [
        {"id": "tokens", "name": "tokens", "op": "Input", "params": {"shape": [1, 4, 16]}, "position": {"x": 0, "y": 0}},
        {"id": "moe", "name": "moe", "op": "MoE", "params": layer, "position": {"x": 1, "y": 0}},
        {"id": "flatten", "name": "flatten", "op": "Flatten", "params": {}, "position": {"x": 2, "y": 0}},
        {"id": "logits", "name": "logits", "op": "Linear", "params": {"out_features": 3}, "position": {"x": 3, "y": 0}},
        {"id": "output", "name": "output", "op": "Output", "params": {}, "position": {"x": 4, "y": 0}},
    ], "edges": [{"id": f"e{i}", "source": a, "target": b} for i, (a, b) in enumerate((("tokens", "moe"), ("moe", "flatten"), ("flatten", "logits"), ("logits", "output")))]}


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class MixtureOfExpertsTests(unittest.TestCase):
    def test_matches_deepseek_v3_routing(self):
        cases = (
            {"score_func": "softmax", "gate_bias": False},
            {"score_func": "sigmoid", "gate_bias": True, "groups": 4, "topk_groups": 2, "route_scale": 2.5},
            {"score_func": "sigmoid", "gate_bias": False, "groups": 2, "topk_groups": 1, "shared_experts": 0},
            {"score_func": "softmax", "gate_bias": True, "shared_experts": 2, "top_k": 3},
        )
        for options in cases:
            with self.subTest(**options):
                torch.manual_seed(3)
                module = TensorLabMoE(16, **{"experts": 8, "top_k": 2, "shared_experts": 1, "expert_dim": 8, **options}).eval()
                if module.gate.bias is not None:
                    # A real bias, so choosing by biased scores and weighing by unbiased ones is tested.
                    module.gate.bias.copy_(torch.randn(8) * 0.3)
                x = torch.randn(2, 5, 16)
                with torch.no_grad():
                    torch.testing.assert_close(module(x), deepseek_moe(module, x.clone()), rtol=1e-5, atol=1e-6)

    def test_every_token_keeps_top_k_distinct_experts(self):
        module = TensorLabMoE(16, experts=8, top_k=3, groups=4, topk_groups=2, gate_bias=True)
        weights, indices = module.gate(torch.randn(10, 16))
        self.assertEqual(tuple(indices.shape), (10, 3))
        self.assertTrue(all(len(set(row.tolist())) == 3 for row in indices))
        # The chosen experts come from at most two of the four groups of two.
        self.assertTrue(all(len({int(e) // 2 for e in row}) <= 2 for row in indices))
        torch.testing.assert_close(weights.sum(-1), torch.ones(10))

    def test_parameter_count_agrees_with_both_analyzers(self):
        for shared in (0, 1, 2):
            with self.subTest(shared=shared):
                module = TensorLabMoE(16, experts=4, top_k=2, shared_experts=shared, expert_dim=8)
                expected = moe_parameters(16, 4, shared, 8)
                self.assertEqual(sum(p.numel() for p in module.parameters()), expected)
                self.assertEqual(analyze_graph(moe_graph({"shared_experts": shared}))["parameters"]["moe"], expected)

    def test_the_gate_and_the_chosen_experts_learn(self):
        model, _ = build_model(moe_graph())
        F.cross_entropy(model(torch.randn(6, 4, 16)), torch.tensor([0, 1, 2, 0, 1, 2])).backward()
        layer = model.layers["moe"]
        self.assertGreater(layer.gate.weight.grad.abs().sum(), 0)
        self.assertTrue(any(expert.w1.weight.grad is not None for expert in layer.experts))
        self.assertGreater(layer.shared_experts.w2.weight.grad.abs().sum(), 0)

    def test_balancing_moves_the_bias_against_the_load(self):
        module = TensorLabMoE(16, experts=4, top_k=1, shared_experts=0, expert_dim=4).train()
        module(torch.randn(32, 16))
        load = module.last_load.float()
        module.balance()
        expected = BALANCE_RATE * torch.sign(load.mean() - load)
        torch.testing.assert_close(module.gate.bias, expected)

    def test_training_balances_only_when_asked(self):
        config = {"epochs": 1, "learningRate": 0.01, "batchSize": 16, "samples": 64, "device": "cpu", "dataset": "synthetic", "validationFraction": 0.2, "earlyStopping": False, "patience": 5}
        for balancing in (False, True):
            with self.subTest(balancing=balancing):
                kept = {}
                def retain(graph, model, preprocessing, metadata):
                    kept["model"] = model
                    return {"modelId": "test"}
                train(moe_graph(), {**config, "moeBalancing": balancing}, lambda message: None, threading.Event(), retain)
                bias = kept["model"].layers["moe"].gate.bias
                self.assertEqual(bool(bias.abs().sum() > 0), balancing)

    def test_imports_the_helper_descriptor(self):
        source = """
from torch import nn
from backend.moe import TensorLabMoE

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = TensorLabMoE(16, experts=4, top_k=2, shared_experts=0, expert_dim=8, score_func="softmax", gate_bias=False)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        return self.head(self.moe(x))
"""
        result = import_pytorch(source, "Model", {"x": [1, 4, 16]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        layer = next(node for node in result["graph"]["nodes"] if node["op"] == "MoE")
        self.assertEqual(layer["params"], {"experts": 4, "top_k": 2, "shared_experts": 0, "expert_dim": 8, "score_func": "softmax", "route_scale": 1.0, "groups": 1, "topk_groups": 1, "gate_bias": 0})
        self.assertEqual(result["analysis"]["totalParameters"], moe_parameters(16, 4, 0, 8) + 17 * 3)
        mismatched = import_pytorch(source, "Model", {"x": [1, 4, 32]})
        self.assertIsNone(mismatched["graph"])


if __name__ == "__main__":
    unittest.main()
