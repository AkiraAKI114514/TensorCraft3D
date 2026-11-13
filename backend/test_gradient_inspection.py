"""Single-sample gradients of the output's cross-entropy; skips without PyTorch."""
import json
import unittest
from unittest.mock import patch

from .inference import InferenceError, TensorInferenceRequest, run_inference
from .training import build_model

try:
    import torch
except ImportError:  # pragma: no cover - deployment without training extra
    torch = None


def chain(*specs):
    nodes = [{"id": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
    return {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(len(nodes) - 1)]}


SAMPLE = [0.5, -1.0, 2.0, 0.25]


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class GradientInspectionTests(unittest.TestCase):
    @staticmethod
    def classifier():
        return chain(("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 3}), ("Tanh", {}), ("Linear", {"out_features": 2}), ("Output", {}))

    def assertClose(self, got, want, places=6):
        self.assertEqual(len(got), len(want))
        for left, right in zip(got, want):
            self.assertAlmostEqual(left, right, places=places)

    def test_the_gradients_are_those_of_the_cross_entropy_against_the_chosen_class(self):
        graph = self.classifier()
        request = dict(graph=graph, nodeIds=["n0", "n1", "n4"], parameters="n3", seed=5, inputs={"n0": SAMPLE})
        report = run_inference(TensorInferenceRequest(**request, gradient={"target": 1}))
        gradient = report["gradient"]
        # The same seeded model and sample through plain autograd.
        torch.manual_seed(5)
        model, _ = build_model(graph)
        model.eval()
        x = torch.tensor([SAMPLE], requires_grad=True)
        kept = {}
        def keep(key, value):
            if key == "n1":
                value.retain_grad()
                kept[key] = value
        logits = model(x, observer=keep)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss.backward()
        probabilities = torch.softmax(logits.detach(), dim=-1)[0]
        self.assertEqual((gradient["target"], gradient["classes"], gradient["predicted"]), (1, 2, int(probabilities.argmax())))
        self.assertAlmostEqual(gradient["loss"], loss.item(), places=6)
        self.assertAlmostEqual(gradient["probability"], float(probabilities[1]), places=6)
        by_node = {entry["nodeId"]: entry for entry in gradient["tensors"]}
        self.assertEqual(set(by_node), {"n0", "n1", "n4"})
        # At the logits the gradient is softmax minus the one-hot target.
        self.assertClose(by_node["n4"]["slice"]["values"][0], (probabilities - torch.tensor([0.0, 1.0])).tolist())
        self.assertClose(by_node["n1"]["slice"]["values"][0], kept["n1"].grad[0].tolist())
        # A floating-point input gets a gradient too.
        self.assertClose(by_node["n0"]["slice"]["values"][0], x.grad[0].tolist())
        self.assertEqual([entry["name"] for entry in gradient["parameters"]], ["weight", "bias"])
        for got, want in zip(gradient["parameters"][0]["slice"]["values"], model.layers["n3"].weight.grad.tolist()):
            self.assertClose(got, want)
        # The values themselves are what an inference-mode run reports; nothing else changes.
        plain = run_inference(TensorInferenceRequest(**request))
        self.assertEqual((report["tensors"], report["parameters"]), (plain["tensors"], plain["parameters"]))
        self.assertIsNone(plain["gradient"])
        json.dumps(report, allow_nan=False)

    def test_integer_ids_have_no_gradient_but_the_vectors_they_select_do(self):
        graph = chain(("Input", {"shape": [1, 3]}), ("Embedding", {"num_embeddings": 5, "embedding_dim": 4}), ("Flatten", {}), ("Linear", {"out_features": 2}), ("Output", {}))
        report = run_inference(TensorInferenceRequest(graph=graph, nodeIds=["n0", "n1", "n4"], gradient={"target": 0}, inputs={"n0": [1, 4, 1]}))
        self.assertEqual([entry["nodeId"] for entry in report["gradient"]["tensors"]], ["n1", "n4"])
        self.assertEqual(report["gradient"]["tensors"][0]["shape"], [1, 3, 4])

    def test_eval_mode_keeps_dropout_off_so_the_gradient_is_repeatable(self):
        graph = chain(("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 8}), ("Dropout", {"p": 0.9}), ("Linear", {"out_features": 3}), ("Output", {}))
        runs = [run_inference(TensorInferenceRequest(graph=graph, nodeIds=["n1"], gradient={"target": 2}, seed=seed, inputs={"n0": SAMPLE}))["gradient"] for seed in (3, 3)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(sum(value == 0 for value in runs[0]["tensors"][0]["slice"]["values"][0]), 0)

    def test_shared_weights_hold_the_gradient_of_every_use(self):
        graph = chain(("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 4}), ("Tanh", {}), ("Linear", {"out_features": 4}), ("Linear", {"out_features": 2}), ("Output", {}))
        graph["nodes"][3]["shares"] = "n1"
        gradient = run_inference(TensorInferenceRequest(graph=graph, nodeIds=["n3"], parameters="n3", gradient={"target": 0}, seed=2, inputs={"n0": SAMPLE}))["gradient"]
        self.assertEqual(gradient["parameters"][0]["owner"], "n1")
        torch.manual_seed(2)
        model, _ = build_model(graph)
        model.eval()
        torch.nn.functional.cross_entropy(model(torch.tensor([SAMPLE])), torch.tensor([0])).backward()
        for got, want in zip(gradient["parameters"][0]["slice"]["values"], model.layers["n1"].weight.grad.tolist()):
            self.assertClose(got, want)

    def test_only_classification_logits_within_the_budget_have_a_gradient(self):
        with self.assertRaisesRegex(InferenceError, "classification logits"):
            run_inference(TensorInferenceRequest(graph=chain(("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 1}), ("Output", {})), nodeIds=["n1"], gradient={"target": 0}))
        with self.assertRaisesRegex(InferenceError, "classification logits"):
            run_inference(TensorInferenceRequest(graph=chain(("Input", {"shape": [1, 5, 4]}), ("Linear", {"out_features": 3}), ("Output", {})), nodeIds=["n1"], gradient={"target": 0}))
        with self.assertRaisesRegex(InferenceError, r"\[0,1\]"):
            run_inference(TensorInferenceRequest(graph=self.classifier(), nodeIds=["n1"], gradient={"target": 2}))
        with self.assertRaises(ValueError):
            TensorInferenceRequest(graph=self.classifier(), allNodes=True, gradient={"target": 0})
        # The backward pass keeps every activation: checked before the model is built.
        with patch("backend.inference.MAX_CAPTURE_BYTES", 100), patch("backend.training.build_model") as build, self.assertRaisesRegex(InferenceError, "Gradient inspection exceeds 64 MB"):
            run_inference(TensorInferenceRequest(graph=self.classifier(), nodeIds=["n1"], gradient={"target": 0}))
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
