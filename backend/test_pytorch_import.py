import ast
import unittest
from unittest.mock import patch
import torch
from torch import nn
from .pytorch_import import import_pytorch
from .training import build_model


def module_source(init, forward, name="Net", inputs="x"):
    return f"from torch import nn\nimport torch\nimport torch.nn.functional as F\nclass {name}(nn.Module):\n    def __init__(self):\n        super().__init__()\n" + "".join(f"        {line}\n" for line in init.splitlines()) + f"    def forward(self, {inputs}):\n" + "".join(f"        {line}\n" for line in forward.splitlines())


class PyTorchImportTests(unittest.TestCase):
    def parsed(self, code, **kwargs):
        result = import_pytorch(code, **kwargs)
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        return result, build_model(result["graph"])[0]

    def reject(self, code, expected=None, **kwargs):
        result = import_pytorch(code, **kwargs)
        self.assertIsNone(result["graph"], result)
        self.assertEqual(result["diagnostics"][0]["level"], "error")
        if expected: self.assertEqual(result["diagnostics"][0]["code"], expected)
        return result

    def test_sequential_equivalence_and_gradients(self):
        code = "from torch import nn\nWIDTH = 4\nmodel = nn.Sequential(nn.Linear(WIDTH, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 2))"
        result, imported = self.parsed(code)
        original = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 2))
        imported.layers["import_1"].load_state_dict(original[0].state_dict())
        imported.layers["import_4"].load_state_dict(original[3].state_dict())
        imported.eval(); original.eval()
        x = torch.randn(3, 4, requires_grad=True)
        torch.testing.assert_close(imported(x), original(x))
        imported(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(result["inputs"][0]["shape"], [1, 4])
        self.assertEqual(result["analysis"]["totalParameters"], sum(p.numel() for p in original.parameters()))

    def test_residual_cnn_and_channels_are_preserved(self):
        code = module_source("self.stem = nn.Conv2d(3, 8, 3, padding=1)\nself.block = nn.Conv2d(8, 8, 3, padding=1)\nself.pool = nn.AdaptiveAvgPool2d((1, 1))\nself.head = nn.Linear(8, 2)", "x = F.relu(self.stem(x))\nresidual = x\nx = F.relu(self.block(x) + residual)\nx = self.pool(x)\nx = x.view(x.size(0), -1)\nreturn self.head(x)")
        result, model = self.parsed(code, input_shapes={"x": [2, 3, 16, 16]})
        self.assertEqual([n["op"] for n in result["graph"]["nodes"]], ["Input", "Conv2d", "ReLU", "Conv2d", "Add", "ReLU", "AdaptiveAvgPool2d", "Flatten", "Linear", "Output"])
        add = next(n for n in result["graph"]["nodes"] if n["op"] == "Add")
        self.assertEqual(len([e for e in result["graph"]["edges"] if e["target"] == add["id"]]), 2)
        output = model(torch.randn(2, 3, 16, 16)); output.sum().backward()
        self.assertEqual(list(output.shape), [2, 2])
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_static_modulelist_nested_custom_defaults_and_model_selection(self):
        code = '''from torch import nn
class Block(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.fc = nn.Linear(width, width)
    def forward(self, x):
        return self.fc(x) + x
class Net(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.block = Block(width)
        self.layers = nn.ModuleList([nn.ReLU(), nn.Linear(width, 2)])
    def forward(self, x):
        x = self.block(x)
        for layer in self.layers:
            x = layer(x)
        return x
model = Net(width=8)
'''
        result, model = self.parsed(code)
        self.assertEqual(result["model"], "Net")
        self.assertEqual(result["models"], ["Block", "Net"])
        self.assertEqual(list(model(torch.randn(2, 8)).shape), [2, 2])
        block, _ = self.parsed(code, model_name="Block")
        self.assertEqual(block["inputs"][0]["shape"], [1, 4])

    def copy_attention(self, target, original):
        branch = target.branches[0]
        e = original.embed_dim
        for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
            getattr(branch, name).load_state_dict({"weight": original.in_proj_weight[i*e:(i+1)*e], "bias": original.in_proj_bias[i*e:(i+1)*e]})
        branch.out_proj.load_state_dict(original.out_proj.state_dict())

    def test_cross_attention_with_distinct_key_and_value(self):
        code = module_source("self.attention = nn.MultiheadAttention(8, 2, batch_first=True)", "x, _ = self.attention(query, key, value, need_weights=False)\nreturn x", inputs="query, key, value")
        result, model = self.parsed(code, input_shapes={"query": [1, 3, 8], "key": [1, 5, 8], "value": [1, 5, 8]})
        original = nn.MultiheadAttention(8, 2, batch_first=True)
        self.copy_attention(model.layers["import_3"], original)
        model.eval(); original.eval()
        q, k, v = [torch.randn(1, size, 8, requires_grad=True) for size in (3, 5, 5)]
        values = {i["id"]: t for i, t in zip(result["inputs"], (q, k, v))}
        torch.testing.assert_close(model(values), original(q, k, v, need_weights=False)[0])
        model(values).square().sum().backward()
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v)))
        self.assertEqual(result["graph"]["nodes"][3]["params"]["attention_type"], "cross")
        self.assertEqual(len([e for e in result["graph"]["edges"] if e.get("targetPort", "").startswith("b0:v")]), 2)

    def test_transformer_matches_pytorch_for_both_norm_orders_and_activations(self):
        for pre_norm in (True, False):
            for activation in ("relu", "gelu"):
                with self.subTest(pre_norm=pre_norm, activation=activation):
                    code = module_source(f'self.encoder = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, activation="{activation}", batch_first=True, norm_first={pre_norm})', "return self.encoder(x)")
                    result, model = self.parsed(code, input_shapes={"x": [1, 4, 8]})
                    original = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, activation=activation, batch_first=True, norm_first=pre_norm)
                    target = model.layers["import_1"]
                    self.copy_attention(target.attention, original.self_attn)
                    target.norm1.load_state_dict(original.norm1.state_dict()); target.norm2.load_state_dict(original.norm2.state_dict())
                    target.ffn[0].load_state_dict(original.linear1.state_dict()); target.ffn[3].load_state_dict(original.linear2.state_dict())
                    model.eval(); original.eval()
                    x = torch.randn(2, 4, 8, requires_grad=True)
                    torch.testing.assert_close(model(x), original(x), atol=1e-6, rtol=1e-5)
                    a = torch.autograd.grad(model(x).square().sum(), x)[0]
                    b = torch.autograd.grad(original(x).square().sum(), x)[0]
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
                    self.assertEqual(result["analysis"]["totalParameters"], sum(p.numel() for p in original.parameters()))

    def test_stacked_transformer_and_concat(self):
        code = module_source("self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, 16, batch_first=True), 3)", "a = self.encoder(x)\nreturn torch.cat([a, x], dim=2)")
        result, model = self.parsed(code, input_shapes={"x": [1, 4, 8]})
        self.assertEqual(sum(n["op"] == "Transformer" for n in result["graph"]["nodes"]), 3)
        self.assertEqual(list(model(torch.randn(2, 4, 8)).shape), [2, 4, 16])

    def test_never_executes_uploaded_code_or_imports(self):
        source = "import nonexistent_module\nraise RuntimeError('do not run')\nfrom torch import nn\nmodel=nn.Sequential(nn.Linear(4, 2))"
        with patch("builtins.exec", side_effect=AssertionError("executed uploaded source")), patch("builtins.eval", side_effect=AssertionError("evaluated uploaded source")):
            result = import_pytorch(source)
        self.assertIsNotNone(result["graph"], result["diagnostics"])

    def test_common_activation_import(self):
        result, model = self.parsed(module_source("self.layer = nn.Sigmoid()", "return self.layer(x)"))
        self.assertEqual(result["graph"]["nodes"][1]["op"], "Sigmoid")
        self.assertEqual(list(model(torch.randn(2, 4)).shape), [2, 4])

    def test_errors_keep_source_location_and_no_partial_graph(self):
        for init, forward, expected in [
            ("self.layer = nn.Linear(4, 2, bias=False)", "return self.layer(x)", "UNSUPPORTED"),
            ("self.layer = nn.Linear(4, 4)", "x = self.layer(x)\nreturn self.layer(x)", "SHARED_WEIGHTS"),
            ("self.layer = nn.Linear(4, 4)", "if x.sum() > 0:\n    x = self.layer(x)\nreturn x", "UNSUPPORTED_STATEMENT"),
            ("self.layer = nn.MultiheadAttention(8, 2)", "return self.layer(x, x, x)[0]", "UNSUPPORTED"),
            ("self.layer = nn.MultiheadAttention(8, 2, batch_first=True)", "x, w = self.layer(x, x, x)\nreturn w", "UNSUPPORTED"),
        ]:
            with self.subTest(init=init, forward=forward):
                result = self.reject(module_source(init, forward), expected)
                self.assertIsInstance(result["diagnostics"][0].get("line"), int)

    def test_shape_mismatch_limits_and_syntax(self):
        code = "from torch import nn\nmodel = nn.Sequential(nn.Linear(4, 2))"
        self.reject(code, "SHAPE_MISMATCH", input_shapes={"x": [1, 5]})
        self.reject(code, "INPUT_SHAPE", input_shapes={"x": [1, 0]})
        self.reject("from torch import nn\nmodel=nn.Sequential(nn.Linear(65536,65536))", "STRUCTURE")
        self.reject("a = (", "SYNTAX")
        self.reject("#" * 512001, "LIMIT")
        self.reject(module_source("self.conv = nn.Conv2d(3, 16, (3, 5))", "return self.conv(x)"), "UNSUPPORTED")

    def test_unsqueeze_and_squeeze_functional_keyword_and_method(self):
        source = module_source(
            "",
            "x = torch.unsqueeze(x, dim=1)\n"
            "x = torch.squeeze(x, dim=1)\n"
            "x = x.squeeze(dim=1)\n"
            "x = torch.squeeze(x, dim=(0, 1))\n"
            "return x",
        )
        result, model = self.parsed(source, input_shapes={"x": [2, 4]})
        self.assertEqual(
            [n["op"] for n in result["graph"]["nodes"]],
            ["Input", "Unsqueeze", "Squeeze", "Squeeze", "Squeeze", "Output"],
        )
        output = model(torch.randn(2, 4))
        self.assertEqual(list(output.shape), [2, 4])

    def test_unsqueeze_aliases_and_tensor_class_form(self):
        cases = [
            ("from torch import unsqueeze", "x = unsqueeze(x, dim=1)\nreturn x"),
            ("import torch.functional", "x = torch.functional.unsqueeze(x, dim=1)\nreturn x"),
            ("import torch", "x = torch.Tensor.unsqueeze(x, dim=1)\nreturn x"),
        ]
        for imports, body in cases:
            with self.subTest(imports=imports):
                result, model = self.parsed(module_source("", body).replace("import torch\n", imports + "\n", 1), input_shapes={"x": [2, 4]})
                self.assertEqual(result["graph"]["nodes"][1]["op"], "Unsqueeze")
                self.assertEqual(list(model(torch.randn(2, 4)).shape), [2, 1, 4])

    def test_input_shapes_remain_available_after_failure(self):
        source = module_source("self.fc = nn.Linear(32, 2)", "x = torch.flatten(x, 1)\nreturn self.fc(x)")
        result = self.reject(source, "SHAPE_MISMATCH", input_shapes={"x": [1, 3, 32, 32]})
        self.assertEqual(result["inputs"][0]["name"], "x")
        valid, _ = self.parsed(source, input_shapes={"x": [1, 2, 4, 4]})
        self.assertEqual(valid["analysis"]["totalParameters"], 66)

    def test_unused_layers_are_explicitly_reported(self):
        result, _ = self.parsed(module_source("self.used = nn.Linear(4, 2)\nself.unused = nn.Linear(2, 2)", "return self.used(x)"))
        self.assertTrue(any(d["code"] == "UNUSED_MODULES" for d in result["diagnostics"]))


    def test_static_positional_encoding_unsqueeze_and_cross_attention(self):
        source = '''import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def _encoder(d_model, nhead, num_layers, dim_feedforward, dropout):
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, num_layers=num_layers)

class Net(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.left = nn.Sequential(nn.Linear(4, width), PositionalEncoding(width, 6))
        self.right = nn.Sequential(nn.Linear(5, width), PositionalEncoding(width, 6))
        self.left_encoder = _encoder(width, 2, 1, 16, 0.1)
        self.right_encoder = _encoder(width, 2, 1, 16, 0.1)
        self.attn = nn.MultiheadAttention(width, 2, batch_first=True)
        self.head = nn.Linear(width, 1)
    def forward(self, left, right):
        left = self.left_encoder(self.left(left))
        right = self.right_encoder(self.right(right))
        attended, _ = self.attn(query=left, key=right, value=right, need_weights=False)
        return self.head((left + attended)[:, -1]).squeeze(-1)
'''
        result = import_pytorch(source, input_shapes={"left": [2, 6, 4], "right": [2, 6, 5]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        self.assertFalse(any(d["code"] == "UNSUPPORTED" and "unsqueeze" in d["message"] for d in result["diagnostics"]))
        self.assertEqual(sum(n["op"] == "MultiHeadAttention" for n in result["graph"]["nodes"]), 1)
        self.assertGreaterEqual(sum(n["op"] == "Transformer" for n in result["graph"]["nodes"]), 1)
        self.assertEqual(sum(n["op"] == "ConstantAdd" for n in result["graph"]["nodes"]), 2)
        self.assertEqual(sum(n["op"] == "Select" for n in result["graph"]["nodes"]), 1)
        imported, info = build_model(result["graph"])
        # Execute only this trusted test fixture to obtain an independent reference.
        namespace = {}; exec(source, namespace)
        original = namespace["Net"]().eval(); imported.eval()
        for node in result["graph"]["nodes"]:
            if node["op"] not in ("Linear", "Transformer", "MultiHeadAttention"): continue
            target = imported.layers[node["id"]]
            path = node["name"].removeprefix("self.")
            if node["op"] == "Transformer": path = path.rsplit('.', 1)[0] + '.layers.' + path.rsplit('.', 1)[1]
            reference = original.get_submodule(path)
            if node["op"] == "Linear": target.load_state_dict(reference.state_dict())
            elif node["op"] == "MultiHeadAttention": self.copy_attention(target, reference)
            else:
                self.copy_attention(target.attention, reference.self_attn)
                target.norm1.load_state_dict(reference.norm1.state_dict()); target.norm2.load_state_dict(reference.norm2.state_dict())
                target.ffn[0].load_state_dict(reference.linear1.state_dict()); target.ffn[3].load_state_dict(reference.linear2.state_dict())
        for node in result["graph"]["nodes"]:
            if node["op"] == "ConstantAdd":
                actual = imported.layers[node["id"]].constant
                torch.testing.assert_close(actual, original.left[1].pe, atol=1e-6, rtol=1e-5)
                self.assertGreater(actual.abs().sum().item(), 0)
        for batch, length in ((1, 3), (3, 6)):
            left = torch.randn(batch, length, 4, requires_grad=True); right = torch.randn(batch, length, 5, requires_grad=True)
            output = imported({item["id"]: value for item, value in zip(result["inputs"], (left, right))})
            expected = original(left, right)
            self.assertEqual(list(output.shape), [batch])
            torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
            for actual_grad, expected_grad in zip(torch.autograd.grad(output.square().sum(), (left, right)), torch.autograd.grad(expected.square().sum(), (left, right))):
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-4)
        self.assertEqual(info["totalParameters"], sum(p.numel() for p in original.parameters()))

    def test_odd_width_position_buffers_and_static_arithmetic(self):
        source = module_source(
            'position = torch.arange(0, 9, dtype=torch.float32).unsqueeze(1)\n'
            'div = torch.exp(torch.arange(0, 5, 2, dtype=torch.float32) * (-math.log(10000.0) / 5))\n'
            'pe = torch.zeros(1, 9, 5)\n'
            'pe[0, :, 0::2] = torch.sin(position * div)\n'
            'pe[0, :, 1::2] = torch.cos(position * div[:2])\n'
            'self.register_buffer("pe", pe, persistent=False)',
            'return (x + self.pe[:, :x.shape[1]])[:, -1]',
        ).replace('import torch\n', 'import torch\nimport math\n', 1)
        result, imported = self.parsed(source, input_shapes={"x": [2, 6, 5]})
        namespace = {}; exec(source, namespace); original = namespace["Net"]()
        for batch, length in ((1, 1), (2, 6), (3, 9)):
            x = torch.zeros(batch, length, 5, requires_grad=True)
            torch.testing.assert_close(imported(x), original(x), atol=1e-6, rtol=1e-5)
            imported(x).sum().backward(); self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(result["analysis"]["totalParameters"], 0)
        source = module_source('a = torch.ones(1, 4)\nb = torch.zeros(1, 4)\nself.register_buffer("buffer", (a + b + 2) * 3 - a / 2)', 'return x + self.buffer')
        _, model = self.parsed(source, input_shapes={"x": [2, 4]})
        x = torch.randn(3, 4)
        torch.testing.assert_close(model(x), x + 8.5)

    def test_tensor_indexing_keeps_all_axes_and_real_values(self):
        cases = [('x[:, 1:5:2, -1]', lambda x: x[:, 1:5:2, -1]), ('x[..., 1:4:2]', lambda x: x[..., 1:4:2]), ('x[:, -1, 1:4:2]', lambda x: x[:, -1, 1:4:2]), ('x[:, None, 1:5, -1]', lambda x: x[:, None, 1:5, -1]), ('torch.select(x, dim=1, index=-1)', lambda x: x.select(1, -1)), ('x.select(-1, 2)', lambda x: x.select(-1, 2))]
        for expression, reference in cases:
            with self.subTest(expression=expression):
                result, model = self.parsed(module_source('', 'return ' + expression), input_shapes={"x": [2, 6, 5]})
                x = torch.randn(3, 6, 5, requires_grad=True)
                actual, expected = model(x), reference(x)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), x)[0], torch.autograd.grad(expected.square().sum(), x)[0])
                self.assertEqual(result["analysis"]["shapes"][result["analysis"]["output"]], list(reference(torch.randn(2, 6, 5)).shape))

    def test_unsupported_static_and_index_semantics_fail_with_source_locations(self):
        cases = [('', 'return x[0]'), ('', 'return x[:1]'), ('', 'return x[:, [0, 2]]'), ('', 'return x[:, ::-1]'), ('', 'return x[:, :x.size(2)]'), ('', 'return x[:, :other.size(1)]'), ('self.register_buffer("pe", torch.empty(1, 6, 5))', 'return x + self.pe'), ('self.register_buffer("pe", torch.ones(1, 6, 5, dtype=torch.float64))', 'return x + self.pe'), ('self.register_buffer("pe", torch.ones(1, 6, 5, unexpected=True))', 'return x + self.pe'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe[:, :other.size(1)]'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe[:, :x.size(2)]'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe[:, :x.size(1), :2]'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x * self.pe'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe.real')]
        for init, forward in cases:
            with self.subTest(init=init, forward=forward):
                result = self.reject(module_source(init, forward, inputs="x, other"), input_shapes={"x": [2, 6, 5], "other": [2, 6, 5]})
                self.assertIsInstance(result["diagnostics"][0].get("line"), int)

    def test_static_work_and_allocation_limits_are_bounded(self):
        for init in ('self.register_buffer("pe", torch.zeros(65537))', 'a = torch.ones(257, 1)\nb = torch.ones(1, 257)\nself.register_buffer("pe", a * b)', 'self.register_buffer("pe", torch.zeros(1.5, 4))'):
            result = self.reject(module_source(init, 'return x + self.pe'), input_shapes={"x": [2, 4]})
            self.assertIsInstance(result["diagnostics"][0].get("line"), int)
        init = 'a = torch.ones(1, 8192)\n' + '\n'.join('a = a + 1' for _ in range(65)) + '\nself.register_buffer("pe", a)'
        self.reject(module_source(init, 'return x + self.pe'), input_shapes={"x": [2, 8192]})

    def test_static_buffer_import_does_not_execute_source(self):
        source = module_source('self.register_buffer("pe", torch.ones(1, 4))', 'return (x + self.pe)[:, -1:]')
        with patch("builtins.exec", side_effect=AssertionError("executed uploaded source")), patch("builtins.eval", side_effect=AssertionError("evaluated uploaded source")):
            result = import_pytorch(source, input_shapes={"x": [2, 4]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])


if __name__ == "__main__": unittest.main()
