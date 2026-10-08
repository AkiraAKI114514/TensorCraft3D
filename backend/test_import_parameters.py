import unittest

import torch
from torch import nn

from .pytorch_import import import_pytorch
from .training import build_model



def module_source(init, forward):
    return (
        "from torch import nn\nimport torch\nimport torch.nn.functional as F\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "".join(f"        {line}\n" for line in init.splitlines())
        + "    def forward(self, x):\n"
        + "".join(f"        {line}\n" for line in forward.splitlines())
    )


class ImportParameterContractTests(unittest.TestCase):
    def parsed(self, source, shape):
        result = import_pytorch(source, input_shapes={"x": shape})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        return result, build_model(result["graph"])[0]

    def rejected(self, source, shape):
        result = import_pytorch(source, input_shapes={"x": shape})
        self.assertIsNone(result["graph"], result)
        self.assertEqual(result["diagnostics"][0]["level"], "error")
        self.assertEqual(result["diagnostics"][0]["code"], "UNSUPPORTED")
        self.assertIsInstance(result["diagnostics"][0].get("line"), int)
        return result

    def test_conv_padding_zero_matches_pytorch_and_gradients(self):
        cases = [
            (1, [2, 3, 9], 3, (0,)),
            (2, [2, 3, 9, 9], 3, (0, 0)),
            (3, [2, 3, 7, 7, 7], 3, (0, 0, 0)),
        ]
        for dimensions, shape, kernel, padding in cases:
            with self.subTest(dimensions=dimensions):
                cls = f"Conv{dimensions}d"
                source = module_source(
                    f"self.layer = nn.{cls}(3, 4, {kernel}, padding={padding})",
                    "return self.layer(x)",
                )
                result, imported = self.parsed(source, shape)
                reference = getattr(nn, cls)(3, 4, kernel, padding=padding)
                imported.layers["import_1"].load_state_dict(reference.state_dict())
                left = torch.randn(*shape, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                torch.testing.assert_close(imported(left), reference(right))
                imported(left).square().sum().backward()
                reference(right).square().sum().backward()
                torch.testing.assert_close(left.grad, right.grad)
                self.assertEqual(result["graph"]["nodes"][1]["params"]["padding"], 0)

    def test_conv_default_padding_zero_is_accepted(self):
        source = module_source("self.layer = nn.Conv2d(3, 4, 3)", "return self.layer(x)")
        result, imported = self.parsed(source, [2, 3, 9, 9])
        reference = nn.Conv2d(3, 4, 3)
        imported.layers["import_1"].load_state_dict(reference.state_dict())
        values = torch.randn(2, 3, 9, 9)
        torch.testing.assert_close(imported(values), reference(values))
        self.assertEqual(result["graph"]["nodes"][1]["params"]["padding"], 0)

    def test_avg_pool_nondefault_options_are_rejected_for_modules_and_functionals(self):
        cases = [
            ("AvgPool1d", "ceil_mode=True", [2, 4, 8]),
            ("AvgPool2d", "count_include_pad=False", [2, 4, 8, 8]),
            ("AvgPool3d", "ceil_mode=True", [2, 4, 8, 8, 8]),
            ("AvgPool2d", "divisor_override=2", [2, 4, 8, 8]),
        ]
        for cls, option, shape in cases:
            with self.subTest(cls=cls, option=option, form="module"):
                self.rejected(module_source(f"self.pool = nn.{cls}(2, 2, 0, {option})", "return self.pool(x)"), shape)
            dimensions = cls[-2]
            function = cls.replace("Pool", "_pool").lower()
            with self.subTest(cls=cls, option=option, form="functional"):
                self.rejected(module_source("", f"return F.{function}(x, 2, 2, 0, {option})"), shape)

    def test_adaptive_max_pool_indices_are_rejected_for_modules_and_functionals(self):
        for dimensions, shape in ((1, [2, 4, 8]), (2, [2, 4, 8, 8]), (3, [2, 4, 8, 8, 8])):
            cls = f"AdaptiveMaxPool{dimensions}d"
            function = cls.replace("AdaptiveMaxPool", "adaptive_max_pool").lower()
            with self.subTest(cls=cls, form="module"):
                self.rejected(module_source(f"self.pool = nn.{cls}(1, return_indices=True)", "return self.pool(x)[0]"), shape)
            with self.subTest(cls=cls, form="functional"):
                self.rejected(module_source("", f"return F.{function}(x, 1, True)[0]"), shape)

    def test_explicit_none_pool_stride_preserves_kernel_default(self):
        source = module_source("self.pool = nn.AvgPool2d(3, stride=None)", "return self.pool(x)")
        result, imported = self.parsed(source, [2, 4, 9, 9])
        reference = nn.AvgPool2d(3, stride=None)
        values = torch.randn(2, 4, 9, 9)
        torch.testing.assert_close(imported(values), reference(values))
        self.assertEqual(result["graph"]["nodes"][1]["params"]["stride"], 3)


if __name__ == "__main__":
    unittest.main()
