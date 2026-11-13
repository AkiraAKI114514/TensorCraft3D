"""Importing causal masks from hand-written attention, SDPA and nn.MultiheadAttention."""
import unittest

import torch
from torch import nn

from .pytorch_import import import_pytorch
from .test_pytorch_import import module_source
from .training import build_model

HEADER = '''
import math
import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim=32, heads=4, block=64):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dim = dim
        self.register_buffer("bias", torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.dim, dim=2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
{body}
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
'''

# Each body computes `y` = causal attention over q/k/v, spelled the ways real code spells it.
SPELLINGS = {
    "masked_fill on a sliced tril buffer": '''
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        y = F.softmax(att, dim=-1) @ v''',
    "masked_fill with an inverted boolean": '''
        att = q @ k.transpose(-2, -1) / math.sqrt(C // self.heads)
        att = att.masked_fill(~self.bias[0, 0, :T, :T].bool(), -math.inf)
        y = att.softmax(-1) @ v''',
    "additive -inf triu": '''
        mask = torch.full((T, T), float("-inf"), device=x.device).triu_(1)
        att = q @ k.transpose(-2, -1) / math.sqrt(C // self.heads)
        y = (att + mask).softmax(dim=-1) @ v''',
    "torch.where on a boolean tril": '''
        keep = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = torch.where(keep, q @ k.transpose(-2, -1) / math.sqrt(C // self.heads), float("-inf"))
        y = att.softmax(-1) @ v''',
    "SDPA is_causal": '''
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.1 if self.training else 0, is_causal=True)''',
    "SDPA boolean attn_mask": '''
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(T, T, dtype=torch.bool).tril())''',
    "SDPA additive attn_mask": '''
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=nn.Transformer.generate_square_subsequent_mask(T))''',
}

# DeepSeek-V3's score layout: positions on axes 1 and 3, heads between them.
BSHT = '''
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.size()
        q = self.wq(x).view(bsz, seqlen, self.heads, -1)
        k = self.wk(x).view(bsz, seqlen, self.heads, -1)
        v = self.wv(x).view(bsz, seqlen, self.heads, -1)
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.scale
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu_(1)
        scores = scores + mask.unsqueeze(1)
        return torch.einsum("bsht,bthd->bshd", scores.softmax(dim=-1), v).flatten(2)
'''


def masks(result):
    return [node["params"] for node in result["graph"]["nodes"] if node["op"] == "CausalMask"]


def imported_with_weights(source, shape=(2, 6, 32)):
    """The source's own module (the oracle) and the imported graph carrying its Linear weights."""
    namespace = {}
    exec(source, namespace)  # the test's own source -- never uploaded code
    reference = namespace["Attention"]().eval()
    result = import_pytorch(source, "Attention", {"x": list(shape)})
    if result["graph"] is None: raise AssertionError(result["diagnostics"])
    model = build_model(result["graph"])[0]
    linears = [node["id"] for node in result["graph"]["nodes"] if node["op"] == "Linear"]
    for key, layer in zip(linears, [m for m in reference.modules() if isinstance(m, nn.Linear)]):
        model.layers[key].load_state_dict(layer.state_dict())
    return result, model.eval(), reference


class CausalAttentionImportTests(unittest.TestCase):
    def test_each_spelling_becomes_one_causal_mask_and_matches_torch(self):
        x = torch.randn(2, 6, 32)
        for name, body in SPELLINGS.items():
            result, model, reference = imported_with_weights(HEADER.format(body=body))
            self.assertEqual(masks(result), [{"query_dim": -2, "key_dim": -1}], name)
            torch.testing.assert_close(model(x), reference(x), msg=name)

    def test_later_positions_do_not_change_earlier_outputs(self):
        _, model, _ = imported_with_weights(HEADER.format(body=SPELLINGS["masked_fill on a sliced tril buffer"]))
        x = torch.randn(1, 6, 32)
        changed = x.clone(); changed[:, 4:] = torch.randn(1, 2, 32)
        torch.testing.assert_close(model(x)[:, :4], model(changed)[:, :4])
        self.assertFalse(torch.allclose(model(x)[:, 4:], model(changed)[:, 4:]))

    def test_the_deepseek_score_layout_masks_axes_one_and_three(self):
        result, model, reference = imported_with_weights(BSHT)
        self.assertEqual(masks(result), [{"query_dim": -3, "key_dim": -1}])
        x = torch.randn(2, 6, 32)
        torch.testing.assert_close(model(x), reference(x))

    def test_sdpa_lowers_to_its_textbook_steps(self):
        forward = "q, k, v = self.q(x), self.k(x), self.v(x)\nreturn F.scaled_dot_product_attention(q, k, v, scale=0.5, dropout_p=0.2 if self.training else 0.0)"
        code = module_source("self.q = nn.Linear(8, 8)\nself.k = nn.Linear(8, 8)\nself.v = nn.Linear(8, 4)", forward)
        result = import_pytorch(code, input_shapes={"x": [2, 5, 8]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        chain = [(node["op"], node["params"]) for node in result["graph"]["nodes"] if node["op"] in ("Einsum", "Scale", "CausalMask", "Softmax", "Dropout")]
        self.assertEqual(chain, [("Einsum", {"equation": "abd,acd->abc"}), ("Scale", {"factor": 0.5}), ("Softmax", {"dim": -1}), ("Dropout", {"p": 0.2}), ("Einsum", {"equation": "abc,ace->abe"})])
        self.assertEqual(result["analysis"]["shapes"][result["graph"]["nodes"][-1]["id"]], [2, 5, 4])
        model = build_model(result["graph"])[0].eval()
        layers = {node["name"].split(".")[-1]: node["id"] for node in result["graph"]["nodes"] if node["op"] == "Linear"}
        x = torch.randn(2, 5, 8)
        q, k, v = (model.layers[layers[name]](x) for name in ("q", "k", "v"))
        torch.testing.assert_close(model(x), nn.functional.scaled_dot_product_attention(q, k, v, scale=0.5))

    def test_multihead_attention_with_a_causal_attn_mask_sets_the_causal_flag(self):
        for mask in ("nn.Transformer.generate_square_subsequent_mask(x.size(1))", "torch.ones(6, 6, dtype=torch.bool).triu(1)"):
            forward = f"y, _ = self.attention(x, x, x, attn_mask={mask}, is_causal=True, need_weights=False)\nreturn y"
            result = import_pytorch(module_source("self.attention = nn.MultiheadAttention(8, 2, batch_first=True)", forward), input_shapes={"x": [2, 6, 8]})
            self.assertIsNotNone(result["graph"], (mask, result["diagnostics"]))
            attention = next(node for node in result["graph"]["nodes"] if node["op"] == "MultiHeadAttention")
            self.assertEqual(attention["params"]["causal"], 1, mask)
            original = nn.MultiheadAttention(8, 2, batch_first=True).eval()
            model = build_model(result["graph"])[0].eval()
            branch = model.layers[attention["id"]].branches[0]
            for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
                getattr(branch, name).load_state_dict({"weight": original.in_proj_weight[i * 8:(i + 1) * 8], "bias": original.in_proj_bias[i * 8:(i + 1) * 8]})
            branch.out_proj.load_state_dict(original.out_proj.state_dict())
            x = torch.randn(2, 6, 8)
            expected = original(x, x, x, attn_mask=nn.Transformer.generate_square_subsequent_mask(6), is_causal=True, need_weights=False)[0]
            torch.testing.assert_close(model(x), expected, msg=mask)

    def test_masks_it_cannot_represent_are_refused(self):
        def rejected(body, fragment):
            result = import_pytorch(HEADER.format(body=body), "Attention", {"x": [2, 6, 32]})
            self.assertIsNone(result["graph"], body)
            self.assertIn(fragment, result["diagnostics"][0]["message"], body)
        scores = "        att = q @ k.transpose(-2, -1)\n"
        rejected(scores + "        y = att.masked_fill(self.bias[:, :, :T, :T] == 0, -1e9).softmax(-1) @ v", "只支持用 -inf 填充")
        rejected(scores + "        y = att.masked_fill(self.bias[:, :, :T, :T], float('-inf')).softmax(-1) @ v", "需要布尔掩码")
        rejected(scores + "        y = att.masked_fill(self.bias[:, :, :T, :T] == 1, float('-inf')).softmax(-1) @ v", "遮住的不是之后的位置")
        rejected(scores + "        y = att.masked_fill(self.bias == 0, float('-inf')).softmax(-1) @ v", "等长且与掩码一致")
        rejected(scores + "        y = att.masked_fill(torch.ones(T, T).tril(-1) == 0, float('-inf')).softmax(-1) @ v", "只支持标准因果掩码")
        rejected(scores + "        y = att.masked_fill(att > 0, float('-inf')).softmax(-1) @ v", "比较运算只支持因果掩码")
        rejected(scores + "        y = (att + self.bias[:, :, :T, :T]).softmax(-1) @ v", "-inf/0 的形式")
        rejected(scores + "        y = torch.tril(att).softmax(-1) @ v", "只支持作用在 torch.ones 或 torch.full")
        rejected(scores + "        y = att.masked_fill(self.bias[:, :, :T, :T - 1] == 0, float('-inf')).softmax(-1) @ v", "等长且与掩码一致")
        rejected("        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)", "推理时也生效")
        rejected("        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, attn_mask=torch.ones(T, T, dtype=torch.bool).tril())", "不能同时给出")
        rejected("        y = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(T, T, dtype=torch.bool).triu(1))", "attn_mask 只支持因果掩码")
        rejected("        y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)", "enable_gqa")
        rejected("        y = (q.float() @ k.transpose(-2, -1)).softmax(-1) @ v", "类型转换只用于因果掩码")

    def test_multihead_attention_masks_it_cannot_represent_are_refused(self):
        for forward, fragment in (
            ("y, _ = self.attention(x, x, x, is_causal=True)\nreturn y", "需要同时给出 attn_mask"),
            ("y, _ = self.attention(x, x, x, attn_mask=torch.ones(6, 6, dtype=torch.bool).tril())\nreturn y", "attn_mask 只支持因果掩码"),
            ("y, _ = self.attention(x, x, x, attn_mask=nn.Transformer.generate_square_subsequent_mask(5))\nreturn y", "与序列长度 6 不一致"),
            ("y, _ = self.attention(x, x, x, key_padding_mask=x[:, :, 0] > 0)\nreturn y", ""),
        ):
            result = import_pytorch(module_source("self.attention = nn.MultiheadAttention(8, 2, batch_first=True)", forward), input_shapes={"x": [2, 6, 8]})
            self.assertIsNone(result["graph"], forward)
            self.assertIn(fragment, result["diagnostics"][0]["message"], forward)

    def test_static_powers_import_and_complex_ones_do_not(self):
        result = import_pytorch(module_source("self.proj = nn.Linear(16, 16)\nself.scale = 16 ** -0.5", "return self.proj(x) * self.scale"), input_shapes={"x": [2, 16]})
        self.assertEqual([node["params"] for node in result["graph"]["nodes"] if node["op"] == "Scale"], [{"factor": 0.25}])
        result = import_pytorch(module_source("self.proj = nn.Linear(16, 16)\nself.scale = (-8) ** 0.5", "return self.proj(x) * self.scale"), input_shapes={"x": [2, 16]})
        self.assertIsNone(result["graph"])

    def test_exported_causal_mask_reimports(self):
        code = module_source("self.q = nn.Linear(8, 8)\nself.k = nn.Linear(8, 8)\nself.mask = TensorLabCausalMask(query_dim=-2, key_dim=-1)", "return self.mask(self.q(x) @ self.k(x).transpose(-2, -1)).softmax(-1)")
        result = import_pytorch(code, input_shapes={"x": [2, 5, 8]})
        self.assertEqual(masks(result), [{"query_dim": -2, "key_dim": -1}])


if __name__ == "__main__":
    unittest.main()
