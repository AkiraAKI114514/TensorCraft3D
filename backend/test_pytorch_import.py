import ast
import unittest
from unittest.mock import patch
import torch
from torch import nn
from .pytorch_import import import_pytorch
from .training import build_model


def module_source(init, forward, name="Net", inputs="x"):
    return f"from torch import nn\nimport torch\nimport torch.nn.functional as F\nimport math\nclass {name}(nn.Module):\n    def __init__(self):\n        super().__init__()\n" + "".join(f"        {line}\n" for line in init.splitlines()) + f"    def forward(self, {inputs}):\n" + "".join(f"        {line}\n" for line in forward.splitlines())


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

    def test_bilinear_infers_each_input_feature_width(self):
        for width in (16, 7):
            for activation in (False, True):
                with self.subTest(width=width, activation=activation):
                    forward = "return self.layer(x, F.relu(y))" if activation else "return self.layer(x, y)"
                    code = module_source(f"self.layer = nn.Bilinear(8, {width}, 10)", forward, inputs="x, y")
                    result, model = self.parsed(code)
                    self.assertEqual([i["shape"] for i in result["inputs"]], [[1, 8], [1, width]])
                    self.assertEqual(result["analysis"]["totalParameters"], 8 * width * 10 + 10)
                    original = nn.Bilinear(8, width, 10)
                    model.layers["import_3" if activation else "import_2"].load_state_dict(original.state_dict())
                    x, y = torch.randn(3, 8), torch.randn(3, width)
                    values = {i["id"]: value for i, value in zip(result["inputs"], (x, y))}
                    torch.testing.assert_close(model(values), original(x, torch.relu(y) if activation else y))
                    self.reject(code, "STRUCTURE", input_shapes={"x": [1, 8], "y": [1, 8]})

    def test_residual_cnn_and_channels_are_preserved(self):
        code = module_source("self.stem = nn.Conv2d(3, 8, 3, padding=1)\nself.block = nn.Conv2d(8, 8, 3, padding=1)\nself.pool = nn.AdaptiveAvgPool2d((1, 1))\nself.head = nn.Linear(8, 2)", "x = F.relu(self.stem(x))\nresidual = x\nx = F.relu(self.block(x) + residual)\nx = self.pool(x)\nx = x.view(x.size(0), -1)\nreturn self.head(x)")
        result, model = self.parsed(code, input_shapes={"x": [2, 3, 16, 16]})
        self.assertEqual([n["op"] for n in result["graph"]["nodes"]], ["Input", "Conv2d", "ReLU", "Conv2d", "Add", "ReLU", "AdaptiveAvgPool2d", "Flatten", "Linear", "Output"])
        add = next(n for n in result["graph"]["nodes"] if n["op"] == "Add")
        self.assertEqual(len([e for e in result["graph"]["edges"] if e["target"] == add["id"]]), 2)
        output = model(torch.randn(2, 3, 16, 16)); output.sum().backward()
        self.assertEqual(list(output.shape), [2, 2])
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_elementwise_gating_imports_as_multiply(self):
        # A GLU/SwiGLU feed-forward: one branch is activated, the other gates it element-wise.
        # Before Multiply existed this failed with "不支持此张量运算", which is what blocked
        # every gated feed-forward in the importer.
        code = module_source(
            "self.gate = nn.Linear(8, 16)\nself.value = nn.Linear(8, 16)\nself.down = nn.Linear(16, 8)",
            "return self.down(F.silu(self.gate(x)) * self.value(x))")
        result, model = self.parsed(code, input_shapes={"x": [2, 4, 8]})
        self.assertEqual([n["op"] for n in result["graph"]["nodes"]], ["Input", "Linear", "SiLU", "Linear", "Multiply", "Linear", "Output"])
        multiply = next(n for n in result["graph"]["nodes"] if n["op"] == "Multiply")
        self.assertEqual(len([e for e in result["graph"]["edges"] if e["target"] == multiply["id"]]), 2)
        output = model(torch.randn(2, 4, 8)); output.sum().backward()
        self.assertEqual(list(output.shape), [2, 4, 8])
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_multiply_rejects_mismatched_shapes(self):
        code = module_source(
            "self.wide = nn.Linear(8, 16)\nself.narrow = nn.Linear(8, 8)",
            "return self.wide(x) * self.narrow(x)")
        self.reject(code, input_shapes={"x": [2, 4, 8]})

    def test_matmul_lowers_to_einsum(self):
        # `a @ b` carries no new IR surface: it becomes an Einsum node with a synthesized
        # equation, so both analyzers keep using the contraction rule they already share.
        code = module_source(
            "self.to_q = nn.Linear(8, 8)\nself.to_k = nn.Linear(8, 8)",
            "return self.to_q(x) @ self.to_k(x).transpose(-2, -1)")
        result, model = self.parsed(code, input_shapes={"x": [2, 4, 8]})
        equations = [n["params"]["equation"] for n in result["graph"]["nodes"] if n["op"] == "Einsum"]
        # Two contractions: the transpose, then the matmul. Equations are verified
        # numerically against `q @ k.transpose(-2, -1)` below rather than by eye.
        self.assertEqual(equations, ["abc->acb", "acb,abd->acd"])
        left, right = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        self.assertTrue(torch.allclose(left @ right.transpose(-2, -1), torch.einsum(equations[1], left, torch.einsum(equations[0], right))))
        output = model(torch.randn(2, 4, 8)); output.sum().backward()
        self.assertEqual(list(output.shape), [2, 4, 4])
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_matmul_spellings_agree(self):
        # torch.matmul / torch.bmm / Tensor.matmul must resolve the same way as `@`;
        # dispatching these on the attribute before the path would misread `torch` as a tensor.
        for forward in (
            "return torch.matmul(self.q(x), self.k(x).transpose(-2, -1))",
            "return self.q(x).matmul(self.k(x).transpose(-2, -1))",
        ):
            code = module_source("self.q = nn.Linear(8, 8)\nself.k = nn.Linear(8, 8)", forward)
            result, _ = self.parsed(code, input_shapes={"x": [2, 4, 8]})
            equations = [n["params"]["equation"] for n in result["graph"]["nodes"] if n["op"] == "Einsum"]
            self.assertEqual(equations, ["abc->acb", "acb,abd->acd"], forward)

    def test_matmul_rejects_mismatched_contraction(self):
        # Both operands are [2,4,8]; the last axis of the left must equal the second-to-last
        # of the right, so this is a real error rather than something to broadcast.
        code = module_source("self.q = nn.Linear(8, 8)\nself.k = nn.Linear(8, 8)", "return self.q(x) @ self.k(x)")
        self.reject(code, input_shapes={"x": [2, 4, 8]})

    def test_bmm_requires_rank_three(self):
        code = module_source("self.q = nn.Linear(8, 8)\nself.k = nn.Linear(8, 8)", "return torch.bmm(self.q(x), self.k(x))")
        self.reject(code, input_shapes={"x": [2, 4, 8]})

    def test_transpose_and_permute_lower_to_einsum(self):
        code = module_source(
            "self.proj = nn.Linear(8, 8)",
            "return self.proj(x).permute(0, 2, 1).transpose(1, 2)")
        result, model = self.parsed(code, input_shapes={"x": [2, 4, 8]})
        # Each op relabels its own input from `abc`, so a swap of the last two axes reads
        # `abc->acb` both times; the compositions cancel and the shape returns to [2,4,8].
        equations = [n["params"]["equation"] for n in result["graph"]["nodes"] if n["op"] == "Einsum"]
        self.assertEqual(equations, ["abc->acb", "abc->acb"])
        self.assertEqual(list(model(torch.randn(2, 4, 8)).shape), [2, 4, 8])

    def test_scalar_scale_imports_as_scale(self):
        # The standard attention scale. `expr` folds math.sqrt and a constructor attribute
        # into one number, so the node carries a concrete factor rather than an expression.
        code = module_source(
            "self.head_dim = 16\nself.to_q = nn.Linear(8, 8)\nself.to_k = nn.Linear(8, 8)",
            "scores = self.to_q(x) @ self.to_k(x).transpose(-2, -1) / math.sqrt(self.head_dim)\n"
            "return scores")
        result, model = self.parsed(code, input_shapes={"x": [2, 4, 8]})
        scales = [n["params"]["factor"] for n in result["graph"]["nodes"] if n["op"] == "Scale"]
        self.assertEqual(scales, [0.25])
        self.assertEqual(list(model(torch.randn(2, 4, 8)).shape), [2, 4, 4])

    def test_scalar_scale_spellings_agree(self):
        # Every one of these must land on the same single-input Scale with the same factor.
        cases = [
            ("return self.p(x) * 2.0", 2.0),
            ("return 2.0 * self.p(x)", 2.0),
            ("return self.p(x) / 4.0", 0.25),
            ("return self.p(x) * (1.0 / math.sqrt(16))", 0.25),
            ("return self.p(x) / math.sqrt(16)", 0.25),
            ("return self.p(x) * -2.0", -2.0),
            ("return self.p(x) * self.scale", 0.5),
        ]
        for forward, expected in cases:
            code = module_source(
                "self.scale = 0.5\nself.p = nn.Linear(8, 8)", forward)
            result, _ = self.parsed(code, input_shapes={"x": [2, 4, 8]})
            scales = [n["params"]["factor"] for n in result["graph"]["nodes"] if n["op"] == "Scale"]
            self.assertEqual(scales, [expected], forward)

    def test_scalar_scale_rejects_non_scale_arithmetic(self):
        # These pair a tensor with a number but are NOT a scale: a reciprocal per element,
        # an affine shift, and a floor. They must keep the generic rejection.
        for forward in (
            "return 2.0 / self.p(x)",
            "return self.p(x) + 2.0",
            "return self.p(x) - 2.0",
        ):
            code = module_source("self.p = nn.Linear(8, 8)", forward)
            self.reject(code, input_shapes={"x": [2, 4, 8]})

    def test_scalar_scale_rejects_division_by_zero(self):
        code = module_source("self.p = nn.Linear(8, 8)", "return self.p(x) / 0.0")
        self.reject(code, input_shapes={"x": [2, 4, 8]})
    def test_biasless_linear_matches_pytorch_parameters_and_values(self):
        # `bias=False` is what every modern LLM/ViT projection writes; it must drop
        # the bias term from both the parameter count and the built module.
        code = module_source("self.layer = nn.Linear(4, 2, bias=False)", "return self.layer(x)")
        result, model = self.parsed(code, input_shapes={"x": [1, 4]})
        self.assertEqual(result["graph"]["nodes"][1]["params"]["bias"], 0)
        layer = model.layers["import_1"]
        original = nn.Linear(4, 2, bias=False)
        layer.load_state_dict(original.state_dict())
        model.eval(); original.eval()
        x = torch.randn(3, 4)
        torch.testing.assert_close(model(x), original(x))
        self.assertIsNone(layer.bias)
        self.assertEqual(result["analysis"]["totalParameters"], 8)
        self.assertEqual(sum(p.numel() for p in original.parameters()), 8)

    def test_biasless_conv_and_bilinear_parameter_counts(self):
        for init, shape, op in [
            ("self.layer = nn.Conv2d(3, 4, 3, padding=1, bias=False)", [1, 3, 8, 8], "Conv2d"),
            ("self.layer = nn.ConvTranspose2d(3, 4, 3, stride=2, bias=False)", [1, 3, 8, 8], "ConvTranspose2d"),
            ("self.a = nn.Linear(4, 6, bias=False)\n"
             "self.b = nn.Linear(4, 6, bias=False)\n"
             "self.layer = nn.Bilinear(6, 6, 5, bias=False)",
             [1, 4], "Bilinear"),
        ]:
            with self.subTest(op=op):
                forward = "return self.layer(self.a(x), self.b(x))" if op == "Bilinear" else "return self.layer(x)"
                result, model = self.parsed(module_source(init, forward), input_shapes={"x": shape})
                target = model.layers["import_3" if op == "Bilinear" else "import_1"]
                self.assertEqual(result["analysis"]["totalParameters"],
                                 sum(p.numel() for p in model.parameters() if p.requires_grad))
                self.assertIsNone(target.bias)

    def test_glu_lowers_to_a_single_op_and_matches_pytorch(self):
        # The single-projection SwiGLU/GLU FFN: one wide projection split in half and
        # sigmoid-gated against itself. `nn.GLU` and `F.glu` must agree.
        for init, forward in [
            ("self.fc = nn.Linear(8, 16)\nself.glu = nn.GLU(dim=-1)",
             "return self.glu(self.fc(x))"),
            ("self.fc = nn.Linear(8, 16)",
             "return F.glu(self.fc(x), dim=-1)"),
        ]:
            with self.subTest(init=init):
                result, model = self.parsed(module_source(init, forward), input_shapes={"x": [2, 4, 8]})
                nodes = [n["op"] for n in result["graph"]["nodes"]]
                self.assertEqual(nodes.count("GLU"), 1)
                self.assertEqual(result["graph"]["nodes"][-2]["params"]["dim"], -1)
                # The 16-wide projection halves back to 8.
                self.assertEqual(result["analysis"]["shapes"]["import_2"], [2, 4, 8])
                output = model(torch.randn(2, 4, 8)); output.sum().backward()
                self.assertEqual(list(output.shape), [2, 4, 8])
                self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_glu_rejects_odd_axis_and_batch_dimension(self):
        # The importer builds the node, then the shared analyzer rejects the shape:
        # an odd axis cannot be halved, and the batch axis may not be touched.
        for init, forward, shape, message in [
            ("self.fc = nn.Linear(4, 5)", "return F.glu(self.fc(x), dim=-1)", [1, 4], "must be even"),
            ("self.fc = nn.Linear(4, 8)", "return F.glu(self.fc(x), dim=0)", [1, 4], "batch dimension"),
        ]:
            with self.subTest(forward=forward):
                code = module_source(init, forward)
                result = import_pytorch(code, "Net", {"x": shape})
                self.assertIsNone(result["graph"], result)
                self.assertEqual(result["diagnostics"][0]["level"], "error")
                self.assertIn(message, result["diagnostics"][0]["message"])


    def test_einsum_contraction_imports_and_runs(self):
        # Scaled attention written out by hand: both contractions are explicit `->` form.
        code = module_source(
            "self.to_q = nn.Linear(8, 8)\nself.to_k = nn.Linear(8, 8)\nself.to_v = nn.Linear(8, 8)\nself.out = nn.Linear(8, 8)",
            "q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)\n"
            "weights = F.softmax(torch.einsum('bsd,btd->bst', q, k), dim=-1)\n"
            "return self.out(torch.einsum('bst,btd->bsd', weights, v))")
        result, model = self.parsed(code, input_shapes={"x": [2, 4, 8]})
        self.assertEqual([n["op"] for n in result["graph"]["nodes"]].count("Einsum"), 2)
        equations = [n["params"]["equation"] for n in result["graph"]["nodes"] if n["op"] == "Einsum"]
        self.assertEqual(equations, ["bsd,btd->bst", "bst,btd->bsd"])
        output = model(torch.randn(2, 4, 8)); output.sum().backward()
        self.assertEqual(list(output.shape), [2, 4, 8])
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_einsum_rejects_implicit_and_invalid_equations(self):
        # An implicit-output equation infers its result from alphabetical label order,
        # and a repeated output label has no single size the analyzer could report.
        for equation in ("bsd,btd", "bsd,btd->bss", "bsd->bst"):
            code = module_source(
                "self.proj = nn.Linear(8, 8)",
                f"return torch.einsum({equation!r}, self.proj(x), self.proj(x))",
                inputs="x")
            self.reject(code, input_shapes={"x": [2, 4, 8]})

    def test_einsum_rejects_mismatched_contraction_sizes(self):
        # `d` is shared between operands, so unequal widths are a real contraction error
        # rather than something broadcasting could absorb.
        code = module_source(
            "self.wide = nn.Linear(8, 16)\nself.thin = nn.Linear(8, 8)",
            "return torch.einsum('bsd,btd->bst', self.wide(x), self.thin(x))")
        self.reject(code, input_shapes={"x": [2, 4, 8]})

    def test_functional_normalization_matches_the_module_forms(self):
        # F.layer_norm / F.rms_norm / F.group_norm are the spellings real transformer code writes.
        # Each must lower onto the same op as its nn.* counterpart and reproduce that module's
        # forward -- the functional call is affine-free, which is the only form the graph can hold.
        cases = [
            ("F.layer_norm(x, (8,))", "self.f = nn.Linear(8, 8)", "return F.layer_norm(self.f(x), (8,))", [2, 8], "LayerNorm", 72),
            ("F.layer_norm(x, (8,), eps=1e-6)", "self.f = nn.Linear(8, 8)", "return F.layer_norm(self.f(x), (8,), eps=1e-6)", [2, 8], "LayerNorm", 72),
            ("F.layer_norm explicit None weight/bias", "self.f = nn.Linear(8, 8)", "return F.layer_norm(self.f(x), (8,), None, None)", [2, 8], "LayerNorm", 72),
            ("F.rms_norm(x, (8,))", "self.f = nn.Linear(8, 8)", "return F.rms_norm(self.f(x), (8,))", [2, 8], "RMSNorm", 72),
            ("F.rms_norm(x, (8,), eps=1e-4)", "self.f = nn.Linear(8, 8)", "return F.rms_norm(self.f(x), (8,), eps=1e-4)", [2, 8], "RMSNorm", 72),
            ("F.group_norm(x, 2)", "self.f = nn.Conv2d(3, 8, 3, padding=1)", "return F.group_norm(self.f(x), 2)", [2, 3, 8, 8], "GroupNorm", 224),
        ]
        for label, init, forward, shape, op, parameters in cases:
            with self.subTest(label=label):
                code = module_source(init, forward)
                result, imported = self.parsed(code, input_shapes={"x": shape})
                node = next(n for n in result["graph"]["nodes"] if n["op"] == op)
                self.assertEqual(node["params"].get("elementwise_affine", node["params"].get("affine")), 0, label)
                namespace = {}
                exec(code, namespace)
                original = namespace["Net"]()
                for candidate in result["graph"]["nodes"]:
                    if candidate["id"] in imported.layers and candidate["name"].startswith("self."):
                        source = original
                        for part in candidate["name"].split("self.", 1)[1].split("."): source = getattr(source, part)
                        imported.layers[candidate["id"]].load_state_dict(source.state_dict())
                imported.eval(); original.eval()
                x = torch.randn(*shape)
                with torch.no_grad():
                    torch.testing.assert_close(imported({i["id"]: x for i in result["inputs"]}), original(x))
                self.assertEqual(result["analysis"]["totalParameters"], parameters, label)

    def test_functional_interpolate_lowers_onto_upsample(self):
        # F.interpolate needs no argument rewriting: its keyword names already match nn.Upsample's.
        for mode, shape in (("nearest", [2, 3, 8, 8]), ("bilinear", [2, 3, 8, 8]), ("nearest", [2, 3, 8, 8, 8])):
            with self.subTest(mode=mode, shape=shape):
                dims = len(shape) - 2
                kernel = "nn.Conv2d(3, 8, 3, padding=1)" if dims == 2 else "nn.Conv3d(3, 8, 3, padding=1)"
                code = module_source(f"self.f = {kernel}", f"return F.interpolate(self.f(x), scale_factor=2, mode='{mode}')")
                result, imported = self.parsed(code, input_shapes={"x": shape})
                node = next(n for n in result["graph"]["nodes"] if n["op"] == "Upsample")
                self.assertEqual(node["params"]["scale_factor"], 2)
                self.assertEqual(node["params"]["mode"], mode)
                namespace = {}
                exec(code, namespace)
                original = namespace["Net"]()
                for candidate in result["graph"]["nodes"]:
                    if candidate["id"] in imported.layers and candidate["name"].startswith("self."):
                        source = original
                        for part in candidate["name"].split("self.", 1)[1].split("."): source = getattr(source, part)
                        imported.layers[candidate["id"]].load_state_dict(source.state_dict())
                imported.eval(); original.eval()
                x = torch.randn(*shape)
                with torch.no_grad():
                    torch.testing.assert_close(imported({i["id"]: x for i in result["inputs"]}), original(x))

    def test_functional_normalization_rejects_affine_tensors(self):
        # A weight/bias argument is a real tensor the graph has no way to store; dropping it
        # silently would change the computation, so it has to be an error rather than a warning.
        for op, init, forward, shape in (
            ("layer_norm", "self.f = nn.Linear(8, 8)", "return F.layer_norm(self.f(x), (8,), torch.ones(8))", [2, 8]),
            ("rms_norm", "self.f = nn.Linear(8, 8)", "return F.rms_norm(self.f(x), (8,), torch.ones(8))", [2, 8]),
            ("group_norm", "self.f = nn.Conv2d(3, 8, 3, padding=1)", "return F.group_norm(self.f(x), 2, torch.ones(8))", [2, 3, 8, 8]),
        ):
            with self.subTest(op=op):
                result = self.reject(module_source(init, forward), input_shapes={"x": shape})
                self.assertIn("weight/bias", result["diagnostics"][0]["message"])
                self.assertIsInstance(result["diagnostics"][0].get("line"), int)

    def test_functional_normalization_rejects_an_unresolvable_eps_and_missing_groups(self):
        # layer_norm's eps is a plain float whose default is 1e-5, so an explicit None is not a
        # request for a dtype-dependent epsilon -- torch raises TypeError for it. Accepting it
        # would put a null in the node params, which both analyzers pass and build_model then
        # fails on, long after the diagnostic could point at the source line.
        result = self.reject(module_source("self.f = nn.Linear(8, 8)", "return F.layer_norm(self.f(x), (8,), None, None, None)"))
        self.assertEqual(result["diagnostics"][0]["code"], "UNSUPPORTED")
        self.assertIsInstance(result["diagnostics"][0].get("line"), int)
        # num_groups may arrive as a keyword, so the bridge must not assume a positional layout.
        for forward in ("return F.group_norm(self.f(x), 2)", "return F.group_norm(self.f(x), num_groups=2)"):
            with self.subTest(forward=forward):
                code = module_source("self.f = nn.Conv2d(3, 8, 3, padding=1)", forward)
                result, imported = self.parsed(code, input_shapes={"x": [2, 3, 8, 8]})
                node = next(n for n in result["graph"]["nodes"] if n["op"] == "GroupNorm")
                self.assertEqual(node["params"]["num_groups"], 2)
                # An affine-free GroupNorm owns nothing, so the total is the Conv2d's weights alone.
                self.assertEqual(node["params"]["affine"], 0)
                self.assertEqual(result["analysis"]["totalParameters"], 3 * 8 * 3 * 3 + 8)
        # A group_norm with no group count at all is a source error, not an internal crash.
        result = self.reject(module_source("self.f = nn.Conv2d(3, 8, 3, padding=1)", "return F.group_norm(self.f(x))"), input_shapes={"x": [2, 3, 8, 8]})
        self.assertIsInstance(result["diagnostics"][0].get("line"), int)

    def test_functional_interpolate_rejects_size_and_align_corners(self):
        # Neither has a representation: `size` is not a scale factor, and align_corners changes the
        # sampling grid, which the Upsample op does not carry.
        for forward in (
            "return F.interpolate(self.f(x), size=(16, 16), mode='nearest')",
            "return F.interpolate(self.f(x), scale_factor=2, mode='bilinear', align_corners=False)",
        ):
            with self.subTest(forward=forward):
                result = self.reject(module_source("self.f = nn.Conv2d(3, 8, 3, padding=1)", forward), input_shapes={"x": [2, 3, 8, 8]})
                self.assertIsInstance(result["diagnostics"][0].get("line"), int)

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

    def test_chunk_and_split_accept_pytorch_keyword_names(self):
        # The method and functional forms name the split spec differently -- `Tensor.split(split_size=)`
        # against `torch.split(split_size_or_sections=)`. Accepting the wrong one would reject valid
        # source and accept a call torch itself refuses.
        valid = ('return x.split(split_size=3, dim=-1)[0]',
                 'return torch.split(x, split_size_or_sections=3, dim=-1)[0]',
                 'return x.chunk(chunks=3, dim=-1)[0]',
                 'return torch.chunk(x, chunks=3, dim=-1)[0]')
        for forward in valid:
            with self.subTest(forward=forward):
                _, model = self.parsed(module_source('', forward), input_shapes={"x": [2, 3, 9]})
                x = torch.randn(2, 3, 9)
                torch.testing.assert_close(model(x), x[:, :, :3])
        invalid = ('return x.split(split_size_or_sections=3, dim=-1)[0]',
                   'return torch.split(x, split_size=3, dim=-1)[0]')
        for forward in invalid:
            with self.subTest(forward=forward):
                # These are exactly the spellings torch raises TypeError on, verified separately so
                # the pairing cannot drift: a wrong keyword name is a defect in both directions.
                expression = forward.replace('return ', '')
                with self.assertRaises(TypeError): eval(expression, {"x": torch.randn(2, 3, 9), "torch": torch})
                self.reject(module_source('', forward), input_shapes={"x": [2, 3, 9]})

    def test_chunk_and_split_lower_onto_slice(self):
        # `q, k, v = qkv.chunk(3, -1)` is the fused-projection spelling every attention block uses.
        # Both chunk and split are pure slicing on a static axis, so they lower onto the existing
        # Slice op instead of adding a second shape path -- the same move matmul made onto Einsum.
        cases = [
            ('q, k, v = x.chunk(3, -1)\nreturn torch.cat([q, k, v], 2)', lambda x: x),
            ('q, k, v = torch.chunk(x, 3, dim=-1)\nreturn torch.cat([q, k, v], 2)', lambda x: x),
            ('a, b, c = x.split([3, 4, 2], -1)\nreturn torch.cat([a, b, c], 2)', lambda x: x),
            ('a, b, c = torch.split(x, (3, 4, 2), -1)\nreturn torch.cat([a, b, c], 2)', lambda x: x),
            # Unequal tail pieces: chunk(3) on a length-7 axis yields sizes [3, 3, 1].
            ('a, b, c = torch.split(x, 3, -1)\nreturn torch.cat([a, b, c], 2)', lambda x: x),
        ]
        for forward, reference in cases:
            with self.subTest(forward=forward.splitlines()[0]):
                result, model = self.parsed(module_source('', forward), input_shapes={"x": [2, 3, 9]})
                # The pieces are slices of one input, so the graph is slices plus a rejoin.
                ops = [n["op"] for n in result["graph"]["nodes"]]
                self.assertGreaterEqual(ops.count("Slice"), 2)
                self.assertNotIn("Chunk", ops)
                x = torch.randn(2, 3, 9, requires_grad=True)
                torch.testing.assert_close(model(x), reference(x))

    def test_chunk_piece_counts_match_pytorch(self):
        # `torch.chunk` is subtler than it looks: the chunk size is ceil(len/chunks) and the empty
        # tail is dropped, so the requested count is an upper bound. A spelling that assumed
        # exactly `chunks` pieces would silently produce a wrong number of slices.
        import backend.pytorch_import as module
        parser = module.Parser.__new__(module.Parser)
        for length in range(1, 25):
            parser.tensor_shape = lambda tensor, source=None, length=length: [1, length]
            for chunks in range(1, 10):
                with self.subTest(length=length, chunks=chunks):
                    sizes = [end - start for _, start, end in parser.slice_pieces(None, chunks, 1, True, None)]
                    self.assertEqual(sizes, [int(t.numel()) for t in torch.chunk(torch.arange(length), chunks)])
                    sizes = [end - start for _, start, end in parser.slice_pieces(None, chunks, 1, False, None)]
                    self.assertEqual(sizes, [int(t.numel()) for t in torch.split(torch.arange(length), chunks)])

    def test_chunk_and_split_reject_unrepresentable_cuts(self):
        # The batch axis is rejected here rather than left to the Slice analyzer so the diagnostic
        # names chunk/split instead of surfacing as a downstream slicing error.
        for forward in ('p = x.chunk(2, 0)\nreturn p[0]', 'p = x.split(2, 0)\nreturn p[0]'):
            with self.subTest(forward=forward.splitlines()[0]):
                result = self.reject(module_source('', forward), input_shapes={"x": [2, 4, 4]})
                self.assertIn("batch", result["diagnostics"][0]["message"])
        cases = ['p = x.chunk(0, 1)\nreturn p[0]', 'p = x.split(0, 1)\nreturn p[0]',
                 'a, b = x.split([3, 3], -1)\nreturn torch.cat([a, b], 2)',
                 'p = x.chunk(2, x.dim())\nreturn p[0]']
        for forward in cases:
            with self.subTest(forward=forward.splitlines()[0]):
                self.reject(module_source('', forward), input_shapes={"x": [2, 4, 4]})
        # An out-of-range dim must be rejected as such. Wrapping it with `% rank` would accept a
        # call torch raises on, and would misreport dim=4 on a rank-4 axis as the batch axis.
        for dim in (4, -5, 9):
            with self.subTest(dim=dim):
                result = self.reject(module_source('', f'return x.chunk(2, {dim})[0]'), input_shapes={"x": [2, 3, 4]})
                self.assertIn("超出张量维度", result["diagnostics"][0]["message"])
        # A count that exceeds the axis is an upper bound, not an error, but the mismatch between
        # the requested unpack and the pieces PyTorch actually returns is reported precisely.
        result = import_pytorch(module_source('', 'q, k, v = x.chunk(2, -1)\nreturn q + k'), input_shapes={"x": [2, 4, 4]})
        self.assertIsNone(result["graph"])
        self.assertIn("返回 2", result["diagnostics"][0]["message"])
        self.assertEqual(result["diagnostics"][0]["level"], "error")

    def test_unsupported_static_and_index_semantics_fail_with_source_locations(self):
        cases = [('', 'return x[0]'), ('', 'return x[:1]'), ('', 'return x[:, [0, 2]]'), ('', 'return x[:, ::-1]'), ('', 'return x[:, :x.size(2)]'), ('', 'return x[:, :other.size(1)]'), ('self.register_buffer("pe", torch.empty(1, 6, 5))', 'return x + self.pe'), ('self.register_buffer("pe", torch.ones(1, 6, 5, dtype=torch.float64))', 'return x + self.pe'), ('self.register_buffer("pe", torch.ones(1, 6, 5, unexpected=True))', 'return x + self.pe'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe[:, :other.size(1)]'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe[:, :x.size(2)]'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe[:, :x.size(1), :2]'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x * self.pe[:, :x.size(2)]'), ('self.register_buffer("pe", torch.ones(1, 6, 5))', 'return x + self.pe.real')]
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


BLOCK = "class Block(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.norm = nn.LayerNorm(d)\n        self.fc = nn.Linear(d, d)\n    def forward(self, x):\n        return self.fc(self.norm(x))\n"


class ImportFoldTests(unittest.TestCase):
    """Static `ModuleList` loops must come back as one folded block, not N flattened copies.

    The import budget is 128 nodes, so a 48-block model only fits at all if the instances
    collapse into one `repeat`; the analyzer then scales the body's parameters by N, which is
    what keeps the count equal to the real model.
    """

    def imported(self, count, forward="for b in self.blocks:\n    x = b(x)\nreturn x"):
        init = f"self.blocks = nn.ModuleList([Block(8) for _ in range({count})])"
        code = "from torch import nn\nimport torch\n" + BLOCK + module_source(init, forward)
        result = import_pytorch(code, model_name="Net", input_shapes={"x": [2, 8]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        return result

    def test_comprehension_modulelist_folds_to_one_instance(self):
        result = self.imported(48)
        graph, analysis = result["graph"], result["analysis"]
        groups = [n for n in graph["nodes"] if n["op"] == "Group"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat"], 48)
        # 3 top-level nodes: Input, the folded block, Output. The 48 copies of the 2-layer body
        # stay inside the subgraph definition.
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual([n["id"] for n in graph["subgraphs"][groups[0]["subgraph"]]["nodes"]], ["n0", "n1"])
        # LayerNorm(8) = 16 params, Linear(8, 8) = 72, so 88 per block and 88 * 48 overall.
        self.assertEqual(analysis["totalParameters"], 88 * 48)
        self.assertEqual(groups[0]["params"], {})

    def test_written_out_modulelist_folds_when_the_instances_are_identical(self):
        init = "self.blocks = nn.ModuleList([Block(8), Block(8), Block(8)])"
        code = "from torch import nn\nimport torch\n" + BLOCK + module_source(init, "for b in self.blocks:\n    x = b(x)\nreturn x")
        result = import_pytorch(code, model_name="Net", input_shapes={"x": [2, 8]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        groups = [n for n in result["graph"]["nodes"] if n["op"] == "Group"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat"], 3)
        self.assertEqual(result["analysis"]["totalParameters"], 88 * 3)

    def test_written_out_instances_that_differ_stay_separate_and_keep_their_own_counts(self):
        # A differing constructor argument makes the two blocks non-isomorphic, so folding them
        # would misreport the parameter count and they must stay separate.
        block = "class Block(nn.Module):\n    def __init__(self, d, p=0.0):\n        super().__init__()\n        self.norm = nn.LayerNorm(d)\n        self.fc = nn.Linear(d, d)\n        self.drop = nn.Dropout(p)\n    def forward(self, x):\n        return self.drop(self.fc(self.norm(x)))\n"
        forward = "for b in self.blocks:\n    x = b(x)\nreturn x"
        mixed = "self.blocks = nn.ModuleList([Block(8), Block(8, 0.5)])"
        result = import_pytorch("from torch import nn\nimport torch\n" + block + module_source(mixed, forward), model_name="Net", input_shapes={"x": [2, 8]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        self.assertEqual([n for n in result["graph"]["nodes"] if n["op"] == "Group"], [])
        self.assertFalse(result["graph"].get("subgraphs"))

    def test_range_indexed_forward_folds(self):
        result = self.imported(4, "for i in range(4):\n    x = self.blocks[i](x)\nreturn x")
        groups = [n for n in result["graph"]["nodes"] if n["op"] == "Group"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat"], 4)
        self.assertEqual(result["analysis"]["totalParameters"], 88 * 4)

    def test_folded_block_trains_with_one_module_per_instance(self):
        result = self.imported(3)
        model, _ = build_model(result["graph"])
        # Expansion gives each instance its own body operators, and they are genuinely
        # independent modules rather than one shared layer.
        norms = [model.layers[f"import_1/{index}/n0"] for index in range(3)]
        linears = [model.layers[f"import_1/{index}/n1"] for index in range(3)]
        self.assertTrue(all(isinstance(layer, nn.LayerNorm) for layer in norms))
        self.assertEqual(len({layer.weight.data_ptr() for layer in linears}), 3)
        output = model(torch.randn(2, 8))
        self.assertEqual(tuple(output.shape), (2, 8))
        output.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(result["analysis"]["totalParameters"], sum(p.numel() for p in model.parameters()))

    def test_elements_that_depend_on_the_loop_variable_are_rejected(self):
        # `Block(i)` gives each instance a different width, so it cannot stand for N identical
        # copies; it must be refused rather than silently collapsed into one instance.
        block = "class Block(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.fc = nn.Linear(d, d)\n    def forward(self, x):\n        return self.fc(x)\n"
        init = "self.blocks = nn.ModuleList([Block(4 + i) for i in range(4)])"
        code = "from torch import nn\nimport torch\n" + block + module_source(init, "for b in self.blocks:\n    x = b(x)\nreturn x")
        result = import_pytorch(code, model_name="Net", input_shapes={"x": [2, 8]})
        self.assertIsNone(result["graph"])
        self.assertIn("循环变量", result["diagnostics"][0]["message"])

    def test_folded_block_infers_the_input_shape_when_none_is_given(self):
        # The body's declared width has to reach the Input through the instance, otherwise the
        # UI would demand a shape for a model whose dimension is already written down.
        init = "self.blocks = nn.ModuleList([Block(8) for _ in range(3)])"
        code = "from torch import nn\nimport torch\n" + BLOCK + module_source(init, "for b in self.blocks:\n    x = b(x)\nreturn x")
        result = import_pytorch(code, model_name="Net")
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        self.assertEqual(result["inputs"][0]["shape"], [1, 8])
        self.assertTrue(result["inputs"][0]["inferred"])

    def test_nested_folded_containers_each_become_one_block(self):
        # A block that itself holds a folded container. The inner fold moves its body into a
        # subgraph, so the outer fold has to run first for both to survive; folding inner-first
        # used to leave the outer body referencing nodes that were already gone.
        inner = "class Inner(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 8)\n    def forward(self, x):\n        return self.fc(x)\n"
        outer = "class Outer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.inner = nn.ModuleList([Inner() for _ in range(3)])\n    def forward(self, x):\n        for layer in self.inner:\n            x = layer(x)\n        return x\n"
        init = "self.stack = nn.ModuleList([Outer() for _ in range(2)])"
        code = "from torch import nn\nimport torch\n" + inner + outer + module_source(init, "for block in self.stack:\n    x = block(x)\nreturn x")
        result = import_pytorch(code, model_name="Net", input_shapes={"x": [2, 8]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        subgraphs = result["graph"]["subgraphs"]
        self.assertEqual(len(subgraphs), 2)
        # Outer x2, inner x3, one Linear(8, 8) each: 2 * 3 * 72 parameters.
        self.assertEqual(result["analysis"]["totalParameters"], 72 * 3 * 2)
        model, _ = build_model(result["graph"])
        self.assertEqual(result["analysis"]["totalParameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(tuple(model(torch.randn(2, 8)).shape), (2, 8))

    def test_rmsnorm_round_trips_through_import_and_matches_the_analyzer(self):
        code = "from torch import nn\nimport torch\n" + module_source("self.norm = nn.RMSNorm(8)", "return self.norm(x)")
        result = import_pytorch(code, model_name="Net", input_shapes={"x": [2, 8]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        node = next(n for n in result["graph"]["nodes"] if n["op"] == "RMSNorm")
        self.assertEqual(node["params"]["elementwise_affine"], 1)
        # nn.RMSNorm leaves eps=None, which the kernel resolves to finfo(opmath).eps rather than
        # the 1e-6 this importer used to record -- for a float32 graph that is finfo(float32).eps.
        self.assertAlmostEqual(node["params"]["eps"], torch.finfo(torch.float32).eps)
        model = build_model(result["graph"])[0]
        # The resolved eps has to reach PyTorch exactly, or the imported graph and the module it
        # came from normalize with different denominators.
        self.assertEqual(model.layers[node["id"]].eps, torch.finfo(torch.float32).eps)
        layer = model.layers[node["id"]]
        self.assertIsInstance(layer, nn.RMSNorm)
        self.assertFalse(hasattr(layer, "bias"))
        self.assertEqual(result["analysis"]["totalParameters"], sum(p.numel() for p in model.parameters()))


    def test_constant_mul_mirrors_constant_add(self):
        # ConstantMul is ConstantAdd's multiplicative sibling, and it exists for one reason: a
        # rotation cannot be written additively. Both ops therefore share the buffer contract
        # (shape/values/sequence_dim rules and rejections) and differ only in the combining op.
        from backend.graph import analyze_graph
        shape, values = [1, 4, 8], [float(i + 1) / 8 for i in range(32)]

        def build(op, params):
            return {"version": 1, "name": op, "nodes": [
                {"id": "i", "name": "I", "op": "Input", "params": {"shape": [2, 4, 8]}, "position": {"x": 0, "y": 0}},
                {"id": "c", "name": "C", "op": op, "params": params, "position": {"x": 0, "y": 0}},
                {"id": "o", "name": "O", "op": "Output", "params": {}, "position": {"x": 0, "y": 0}}],
                "edges": [{"id": "a", "source": "i", "target": "c"}, {"id": "b", "source": "c", "target": "o"}]}

        add, mul = build("ConstantAdd", {"shape": shape, "values": values, "sequence_dim": 1}), build("ConstantMul", {"shape": shape, "values": values, "sequence_dim": 1})
        self.assertEqual(analyze_graph(add)["shapes"], analyze_graph(mul)["shapes"])
        # Neither buffer is trainable, so neither contributes parameters.
        self.assertEqual(analyze_graph(mul)["totalParameters"], 0)

        x = torch.randn(2, 4, 8)
        buffer = torch.tensor(values).reshape(shape)
        for op, expected in (("ConstantAdd", x + buffer), ("ConstantMul", x * buffer)):
            with self.subTest(op=op):
                model, _ = build_model(build(op, {"shape": shape, "values": values, "sequence_dim": 1}))
                model.eval()
                with torch.no_grad(): torch.testing.assert_close(model({"i": x}), expected)

        # The validation rules are shared, so each rejection must fire identically for both ops.
        for params in ({"shape": [8.5], "values": [1.0] * 8}, {"shape": [1, 8], "values": [1.0] * 7},
                       {"shape": [2, 8], "values": [1.0] * 16}, {"shape": [1, 2, 8], "values": [1.0] * 16, "sequence_dim": 1}):
            for op in ("ConstantAdd", "ConstantMul"):
                with self.subTest(op=op, params=params):
                    with self.assertRaises(ValueError): analyze_graph(build(op, params))

    def test_rotary_position_embedding_is_expressible(self):
        # The point of ConstantMul. RoPE is x*cos + rotate_half(x)*sin; the cos/sin are fixed
        # buffers, so without a multiplicative constant this rotation cannot be written at all.
        # This is the primitive DeepSeek-V3's MLA depends on.
        from backend.graph import analyze_graph
        seq, dim = 4, 8
        angle = torch.arange(seq).float().unsqueeze(1) * (1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))
        cos = torch.cat([angle.cos(), angle.cos()], -1); sin = torch.cat([angle.sin(), angle.sin()], -1)
        rotate_half = lambda v: torch.cat([-v[..., dim // 2:], v[..., :dim // 2]], -1)
        graph = {"version": 1, "name": "RoPE", "nodes": [
            {"id": "in", "name": "In", "op": "Input", "params": {"shape": [2, seq, dim]}, "position": {"x": 0, "y": 0}},
            {"id": "cos", "name": "Cos", "op": "ConstantMul", "params": {"shape": [1, seq, dim], "values": cos.reshape(-1).tolist(), "sequence_dim": 1}, "position": {"x": 0, "y": 0}},
            {"id": "lo", "name": "Lo", "op": "Slice", "params": {"dim": 2, "start": "none", "end": dim // 2, "step": 1}, "position": {"x": 0, "y": 0}},
            {"id": "hi", "name": "Hi", "op": "Slice", "params": {"dim": 2, "start": dim // 2, "end": "none", "step": 1}, "position": {"x": 0, "y": 0}},
            {"id": "neg", "name": "Neg", "op": "Scale", "params": {"factor": -1}, "position": {"x": 0, "y": 0}},
            {"id": "rot", "name": "Rot", "op": "Concat", "params": {"dim": 2}, "position": {"x": 0, "y": 0}},
            {"id": "sin", "name": "Sin", "op": "ConstantMul", "params": {"shape": [1, seq, dim], "values": sin.reshape(-1).tolist(), "sequence_dim": 1}, "position": {"x": 0, "y": 0}},
            {"id": "sum", "name": "Sum", "op": "Add", "params": {}, "position": {"x": 0, "y": 0}},
            {"id": "out", "name": "Out", "op": "Output", "params": {}, "position": {"x": 0, "y": 0}}], "edges": [
            {"id": "e1", "source": "in", "target": "cos"}, {"id": "e2", "source": "in", "target": "lo"},
            {"id": "e3", "source": "in", "target": "hi"}, {"id": "e4", "source": "hi", "target": "neg"},
            {"id": "e5", "source": "neg", "target": "rot"}, {"id": "e6", "source": "lo", "target": "rot"},
            {"id": "e7", "source": "rot", "target": "sin"}, {"id": "e8", "source": "cos", "target": "sum"},
            {"id": "e9", "source": "sin", "target": "sum"}, {"id": "e10", "source": "sum", "target": "out"}]}
        self.assertEqual(analyze_graph(graph)["shapes"]["out"], [2, seq, dim])
        model, _ = build_model(graph); model.eval()
        x = torch.randn(2, seq, dim)
        with torch.no_grad(): torch.testing.assert_close(model({"in": x}), x * cos + rotate_half(x) * sin)

    def test_a_buffer_multiplication_imports_as_constant_mul(self):
        # The multiplicative spelling has to be reachable from the PyTorch source it was added for,
        # not just from a hand-built graph: `x * self.cos` is how RoPE is actually written.
        cos = torch.ones(1, 8, 8) * 1.5
        cos = cos - 0.25
        init = 'pe = torch.ones(1, 8, 8) * 1.5\npe[0] = pe[0] - 0.25\nself.register_buffer("cos", pe)'
        cases = (('return x * self.cos', "ConstantMul", lambda x: x * cos),
                 ('return self.cos * x', "ConstantMul", lambda x: x * cos),
                 ('return x + self.cos', "ConstantAdd", lambda x: x + cos))
        for forward, op, reference in cases:
            with self.subTest(forward=forward):
                result = import_pytorch(module_source(init, forward), input_shapes={"x": [2, 8, 8]})
                self.assertIsNotNone(result["graph"], result["diagnostics"])
                self.assertIn(op, [n["op"] for n in result["graph"]["nodes"]])
                # A buffer is not trainable either way, so both spellings stay parameter-free.
                self.assertEqual(result["analysis"]["totalParameters"], 0)
                model, _ = build_model(result["graph"])
                key = result["inputs"][0]["id"]
                x = torch.randn(2, 8, 8)
                with torch.no_grad(): torch.testing.assert_close(model({key: x}), reference(x))

        # A dynamic sequence prefix narrows exactly as the additive path does.
        result = import_pytorch(module_source(init, 'return x * self.cos[:, :x.size(1)]'), input_shapes={"x": [2, 5, 8]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        node = next(n for n in result["graph"]["nodes"] if n["op"] == "ConstantMul")
        self.assertEqual(node["params"]["sequence_dim"], 1)
        model, _ = build_model(result["graph"])
        key = result["inputs"][0]["id"]
        x = torch.randn(2, 5, 8)
        with torch.no_grad(): torch.testing.assert_close(model({key: x}), x * cos[:, :5])
        # Multiplication against a non-sequence axis is still rejected, mirroring the additive rule.
        rejected = import_pytorch(module_source(init, 'return x * self.cos[:, :x.size(2)]'), input_shapes={"x": [2, 8, 8]})
        self.assertIsNone(rejected["graph"], rejected["graph"])
        self.assertEqual(rejected["diagnostics"][0]["level"], "error")



if __name__ == "__main__": unittest.main()
