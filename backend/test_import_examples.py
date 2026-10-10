"""Static import and runtime contracts for the bundled, trusted examples."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from .pytorch_import import import_pytorch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    torch = None


EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "import_models"
MANIFEST = json.loads((EXAMPLES / "manifest.json").read_text(encoding="utf-8"))["examples"]


def parse_example(example):
    source = (EXAMPLES / example["filename"]).read_text(encoding="utf-8")
    return import_pytorch(source, model_name=example["model_name"], input_shapes=example["input_shapes"])


class ImportExampleTests(unittest.TestCase):
    def test_static_import_matches_manifest_without_executing_source(self):
        self.assertEqual({example["filename"] for example in MANIFEST}, {"cnn.py", "residual.py", "mqa.py", "gqa.py", "cross_attention.py", "grouped_stack.py"})
        for example in MANIFEST:
            with self.subTest(filename=example["filename"]):
                with patch("builtins.exec", side_effect=AssertionError("executed source")), patch("builtins.eval", side_effect=AssertionError("evaluated source")):
                    result = parse_example(example)
                self.assertIsNotNone(result["graph"], result["diagnostics"])
                self.assertEqual(result["model"], example["model_name"])
                self.assertEqual({item["name"]: item["shape"] for item in result["inputs"]}, example["input_shapes"])
                self.assertFalse(any(item["inferred"] for item in result["inputs"]))
                self.assertEqual([node["op"] for node in result["graph"]["nodes"]], example["expected_ops"])
                info = result["analysis"]
                self.assertEqual(info["shapes"][info["output"]], example["expected_output"])
                self.assertEqual(info["totalParameters"], example["expected_parameters"])
                self.assertEqual([item["code"] for item in result["diagnostics"]], ["STRUCTURE_ONLY"])
                if "attention" in example:
                    params = next(node["params"] for node in result["graph"]["nodes"] if node["op"] in ("MultiHeadAttention", "Transformer"))
                    for name, value in example["attention"].items(): self.assertEqual(params[name], value)
                if example["filename"] == "cross_attention.py":
                    attention = next(node for node in result["graph"]["nodes"] if node["op"] == "MultiHeadAttention")
                    self.assertEqual({edge["targetPort"] for edge in result["graph"]["edges"] if edge["target"] == attention["id"]}, {"query", "context"})

    @unittest.skipUnless(torch is not None, "PyTorch not installed; static example imports are still covered")
    def test_graph_matches_trusted_source_with_common_weights_and_gradients(self):
        from .training import build_model
        for example in MANIFEST:
            with self.subTest(filename=example["filename"]):
                torch.manual_seed(42)
                result = parse_example(example)
                self.assertIsNotNone(result["graph"], result["diagnostics"])
                imported, info = build_model(result["graph"])
                # Execute only our checked-in example, never uploaded source.
                spec = importlib.util.spec_from_file_location("import_example_reference", EXAMPLES / example["filename"])
                reference_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(reference_module)
                original = getattr(reference_module, example["model_name"])()
                for node in result["graph"]["nodes"]:
                    if node["id"] not in imported.layers: continue
                    target = imported.layers[node["id"]]
                    try:
                        reference = original.get_submodule(node["name"].removeprefix("self."))
                    except AttributeError:
                        # Functional activations/reshapes are represented by graph
                        # nodes but have no reference module or trainable weights.
                        continue
                    if isinstance(reference, torch.nn.MultiheadAttention):
                        branch, width = target.branches[0], reference.embed_dim
                        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
                            start, end = index * width, (index + 1) * width
                            getattr(branch, name).load_state_dict({"weight": reference.in_proj_weight[start:end], "bias": reference.in_proj_bias[start:end]})
                        branch.out_proj.load_state_dict(reference.out_proj.state_dict())
                    else: target.load_state_dict(reference.state_dict())
                imported.eval(); original.eval()
                self.assertEqual(info["totalParameters"], example["expected_parameters"])
                for model in (imported, original): self.assertEqual(sum(p.numel() for p in model.parameters()), example["expected_parameters"])
                tensors = {name: torch.randn(*shape, requires_grad=True) for name, shape in example["input_shapes"].items()}
                reference_inputs = {name: value.detach().clone().requires_grad_() for name, value in tensors.items()}
                graph_inputs = {item["id"]: tensors[item["name"]] for item in result["inputs"]}
                output = imported(graph_inputs)
                expected = original(**reference_inputs)
                self.assertEqual(list(output.shape), example["expected_output"])
                torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
                output.square().sum().backward(); expected.square().sum().backward()
                for name, value in tensors.items():
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all().item())
                    torch.testing.assert_close(value.grad, reference_inputs[name].grad, atol=1e-6, rtol=1e-4)
                for model in (imported, original):
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))


if __name__ == "__main__": unittest.main()
