"""Importing hand-written attention: shape reads, head split/merge reshapes and softmax spellings."""
import math
import unittest

import torch

from .pytorch_import import import_pytorch
from .test_pytorch_import import module_source, reference_model
from .training import build_model

ATTENTION = '''
import math
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, dim=32, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        {shape}
        q = self.wq(x).{split_q}.transpose(1, 2)
        k = self.wk(x).{split_k}.transpose(1, 2)
        v = self.wv(x).{split_v}.transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        probs = {softmax}
        out = (probs @ v).transpose(1, 2).{merge}
        return self.wo(out)
'''

# Each spelling of the same attention block; all of them must compute the same thing.
SPELLINGS = [
    dict(shape="B, S, _ = x.shape", split_q="view(B, S, self.n_heads, self.head_dim)", split_k="view(B, S, self.n_heads, -1)", split_v="reshape(B, S, self.n_heads, self.head_dim)", softmax="scores.softmax(dim=-1)", merge="contiguous().view(B, S, -1)"),
    dict(shape="bsz, seqlen, _ = x.size()", split_q="view(bsz, seqlen, self.n_heads, -1)", split_k="unflatten(-1, (self.n_heads, self.head_dim))", split_v="view(*x.shape[:-1], self.n_heads, self.head_dim)", softmax="torch.softmax(scores, dim=-1)", merge="flatten(2)"),
    dict(shape="B, S = x.shape[:2]", split_q="view(x.size(0), x.size(1), self.n_heads, self.head_dim)", split_k="view(-1, S, self.n_heads, self.head_dim)", split_v="view(B, S, self.n_heads, self.head_dim)", softmax="scores.softmax(-1)", merge="reshape(B, S, self.n_heads * self.head_dim)"),
]


def lowered(result):
    return [(node["op"], node["params"]) for node in result["graph"]["nodes"] if node["op"] in ("Flatten", "Unflatten")]


class HandWrittenAttentionTests(unittest.TestCase):
    def test_each_spelling_imports_and_matches_torch(self):
        x = torch.randn(2, 6, 32)
        for spelling in SPELLINGS:
            source = ATTENTION.format(**spelling)
            namespace = {}
            exec(source, namespace)  # the test's own source, the oracle -- never uploaded code
            reference = namespace["Attention"]().eval()
            result = import_pytorch(source, "Attention", {"x": [2, 6, 32]})
            self.assertIsNotNone(result["graph"], (spelling, result["diagnostics"]))
            # Three head splits, one merge, nothing else reshaped.
            self.assertEqual(lowered(result), [("Unflatten", {"dim": -1, "sizes": [4, 8]})] * 3 + [("Flatten", {"start_dim": 2, "end_dim": -1})], spelling)
            model, info = build_model(result["graph"])
            self.assertEqual(info["totalParameters"], sum(p.numel() for p in reference.parameters()))
            linears = [node["id"] for node in result["graph"]["nodes"] if node["op"] == "Linear"]
            for key, layer in zip(linears, (reference.wq, reference.wk, reference.wv, reference.wo)):
                model.layers[key].load_state_dict(layer.state_dict())
            model.eval()
            torch.testing.assert_close(model(x), reference(x), msg=str(spelling))

    def test_gradients_flow_through_the_reshapes(self):
        source = ATTENTION.format(**SPELLINGS[0])
        result = import_pytorch(source, "Attention", {"x": [2, 6, 32]})
        model = build_model(result["graph"])[0]
        model(torch.randn(2, 6, 32)).sum().backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))


class ReshapeLoweringTests(unittest.TestCase):
    # (forward, input shape, expected Flatten/Unflatten nodes)
    CASES = [
        ("return x.view(x.size(0), -1)", [2, 3, 4], [("Flatten", {})]),
        ("B, S, E = x.shape\nreturn x.view(B, S, 4, 8)", [2, 6, 32], [("Unflatten", {"dim": -1, "sizes": [4, 8]})]),
        ("B, S, E = x.shape\nreturn x.view(B, 2, 3, E)", [2, 6, 32], [("Unflatten", {"dim": 1, "sizes": [2, 3]})]),
        ("B, S, E = x.shape\nreturn x.reshape(B, S * 4, 8)", [2, 6, 32], [("Flatten", {}), ("Unflatten", {"dim": -1, "sizes": [24, 8]})]),
        ("B, S, E = x.shape\nreturn torch.reshape(x, (B, E, S))", [2, 6, 32], [("Flatten", {}), ("Unflatten", {"dim": -1, "sizes": [32, 6]})]),
        ("return x.flatten(2)", [2, 6, 4, 8], [("Flatten", {"start_dim": 2, "end_dim": -1})]),
        ("return torch.flatten(x, 1, 2)", [2, 6, 4, 8], [("Flatten", {"start_dim": 1, "end_dim": 2})]),
        ("return x.view(x.size(0), x.size(1), -1, x.size(-1))", [2, 6, 4, 8], []),
        ("B, S, H, D = x.shape\nreturn x.view(B, S * H, D)", [2, 6, 4, 8], [("Flatten", {"start_dim": 1, "end_dim": 2})]),
        ("return x.unflatten(1, (2, 3))", [2, 6, 8], [("Unflatten", {"dim": 1, "sizes": [2, 3]})]),
        ("return torch.unflatten(x, -1, (2, -1))", [2, 6, 8], [("Unflatten", {"dim": -1, "sizes": [2, -1]})]),
        ("return x.view(*x.shape, 1)", [2, 6, 8], [("Unflatten", {"dim": -1, "sizes": [8, 1]})]),
        ("B, S, _, E = x.shape\nreturn x.view(B, S, E)", [2, 6, 1, 8], [("Flatten", {"start_dim": 2, "end_dim": -1})]),
        ("return x.contiguous().view(-1, 6, 2, 4)", [2, 6, 8], [("Unflatten", {"dim": -1, "sizes": [2, 4]})]),
    ]

    def test_reshapes_lower_onto_flatten_and_unflatten_exactly(self):
        for forward, shape, expected in self.CASES:
            result = import_pytorch(module_source("self.proj = nn.Linear(%d, %d)" % (shape[-1], shape[-1]), "x = self.proj(x)\n" + forward), input_shapes={"x": shape})
            self.assertIsNotNone(result["graph"], (forward, result["diagnostics"]))
            self.assertEqual(lowered(result), expected, forward)
            model = build_model(result["graph"])[0].eval()
            layer = next(node["id"] for node in result["graph"]["nodes"] if node["op"] == "Linear")
            reference = reference_model(forward)
            x = torch.randn(*shape)
            projected = model.layers[layer](x)
            # A reshape is a row-major reinterpretation; nothing is computed, so the values are equal.
            self.assertTrue(torch.equal(model(x), reference(projected)), forward)

    def test_registered_flatten_and_unflatten_modules(self):
        code = module_source("self.split = nn.Unflatten(-1, (4, 8))\nself.merge = nn.Flatten(2)", "return self.merge(self.split(x))")
        result = import_pytorch(code, input_shapes={"x": [2, 6, 32]})
        self.assertEqual(lowered(result), [("Unflatten", {"dim": -1, "sizes": [4, 8]}), ("Flatten", {"start_dim": 2, "end_dim": -1})])
        self.assertEqual(result["analysis"]["shapes"][result["graph"]["nodes"][-1]["id"]], [2, 6, 32])

    def test_reshapes_that_would_lose_the_batch_axis_are_refused(self):
        for forward, fragment in (
            ("return x.view(2, 6, 4, 8)", "第一维需要保留批次"),
            ("return x.view(-1, 32)", "并入其他维度"),
            ("B, S, E = x.shape\nreturn x.view(B * S, E)", "批次大小不能参与运算"),
            ("B, S, E = x.shape\nreturn x.view(S, B, E)", "只能作为 view/reshape 的第一维"),
            ("B, S, E = x.shape\nreturn x.view(B, 5, -1)", "无法把非批次维度"),
            ("B, S, E = x.shape\nreturn x.view(B, S, 4, 9)", "元素数与输入不一致"),
            ("B, S, E = x.shape\nreturn x.view(B, -1, -1)", "最多只能有一个 -1"),
            ("B, S, E = x.shape\nreturn x.view(B, S, E / 4, 4)", "正整数、-1 或批次大小"),
            ("return x.flatten(0)", "start_dim 需要从 1 开始"),
            ("return x.unflatten(0, (1, 2))", "不能拆分批次维度"),
            ("return x.softmax()", "需要 dim"),
            ("return x.size(3)", "维度越界"),
        ):
            result = import_pytorch(module_source("", forward), input_shapes={"x": [2, 6, 32]})
            self.assertIsNone(result["graph"], forward)
            self.assertIn(fragment, result["diagnostics"][0]["message"], forward)

    def test_a_negative_start_on_the_batch_axis_is_caught_by_the_analysis(self):
        result = import_pytorch(module_source("", "return x.flatten(-3)"), input_shapes={"x": [2, 6, 32]})
        self.assertIsNone(result["graph"])
        self.assertEqual(result["diagnostics"][0]["code"], "STRUCTURE")
        self.assertEqual(result["diagnostics"][0]["line"], 9)

    def test_reading_a_shape_needs_the_input_shape(self):
        result = import_pytorch(module_source("", "B, S, E = x.shape\nreturn x.view(B, S, 4, -1)"))
        self.assertIsNone(result["graph"])
        self.assertEqual(result["diagnostics"][0]["code"], "INPUT_SHAPE")

    def test_softmax_spellings_agree(self):
        for forward in ("return x.softmax(dim=-1)", "return x.softmax(-1)", "return torch.softmax(x, dim=-1)", "return F.softmax(x, dim=-1)", "return x.log_softmax(-1)", "return torch.log_softmax(x, -1)"):
            result = import_pytorch(module_source("", forward), input_shapes={"x": [2, 6, 32]})
            self.assertIsNotNone(result["graph"], (forward, result["diagnostics"]))
            model, x = build_model(result["graph"])[0], torch.randn(2, 6, 32)
            torch.testing.assert_close(model(x), reference_model(forward)(x), msg=forward)


if __name__ == "__main__":
    unittest.main()
