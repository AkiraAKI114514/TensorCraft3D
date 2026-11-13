"""One output value explained by the measured terms it is made of; skips without PyTorch."""
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


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class ValueExplanationTests(unittest.TestCase):
    @staticmethod
    def explain(graph, node, index, **kwargs):
        report = run_inference(TensorInferenceRequest(graph=graph, nodeIds=[node], explain={"nodeId": node, "index": index}, **kwargs))
        return report, report["explanation"]

    def assertAddsUp(self, explanation):
        # The float64 terms rebuild the measured float32 value, and the listed terms, the summed
        # rest and the bias are exactly what they rebuild it from.
        self.assertAlmostEqual(explanation["recomputed"], explanation["value"], places=5)
        if explanation["rule"] != "max":
            parts = sum(term["term"] for term in explanation["terms"]) + explanation["rest"]["sum"] + (explanation["bias"] or 0.0)
            self.assertAlmostEqual(parts, explanation["recomputed"], places=9)

    def test_a_linear_output_is_its_inputs_times_one_weight_row_plus_the_bias(self):
        graph = chain(("Input", {"shape": [1, 40]}), ("Linear", {"out_features": 3}), ("Output", {}))
        sample = [0.1 * (i - 20) for i in range(40)]
        report, explanation = self.explain(graph, "n1", [2], seed=3, inputs={"n0": sample})
        self.assertEqual(explanation["value"], report["tensors"][0]["slice"]["values"][0][2])
        self.assertEqual((explanation["rule"], explanation["count"], explanation["padded"], explanation["inputNode"], explanation["inputShape"]), ("weighted-sum", 40, 0, "n0", [40]))
        self.assertEqual((len(explanation["terms"]), explanation["rest"]["count"]), (16, 24))
        magnitudes = [abs(term["term"]) for term in explanation["terms"]]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        # Each term is the sample's value times the weight the forward pass multiplied it by.
        torch.manual_seed(3)
        model, _ = build_model(graph)
        weight, bias = model.layers["n1"].weight.detach(), model.layers["n1"].bias.detach()
        for term in explanation["terms"]:
            feature = term["input"][0]
            self.assertEqual(term["parameter"], [2, feature])
            self.assertAlmostEqual(term["x"], sample[feature], places=6)
            self.assertAlmostEqual(term["w"], float(weight[2, feature]), places=7)
            self.assertAlmostEqual(term["term"], term["x"] * term["w"], places=12)
        self.assertAlmostEqual(explanation["bias"], float(bias[2]), places=7)
        self.assertAddsUp(explanation)

    def test_a_linear_layer_over_a_sequence_reads_the_position_of_the_value(self):
        graph = chain(("Input", {"shape": [1, 5, 8]}), ("Linear", {"out_features": 3}), ("Output", {}))
        _, explanation = self.explain(graph, "n1", [4, 1])
        self.assertEqual({tuple(term["input"][:1]) for term in explanation["terms"]}, {(4,)})
        self.assertEqual({tuple(term["parameter"][:1]) for term in explanation["terms"]}, {(1,)})
        self.assertAddsUp(explanation)

    def test_a_convolution_output_reads_its_window_across_its_group_s_input_channels(self):
        graph = chain(("Input", {"shape": [1, 4, 6, 6]}), ("Conv2d", {"out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 1, "groups": 2}), ("Output", {}))
        # At the corner the 3 x 3 window hangs over the top and left edges, which read zeros.
        _, corner = self.explain(graph, "n1", [5, 0, 0])
        self.assertEqual((corner["count"], corner["padded"]), (2 * 9, 2 * 5))
        # Output channel 5 is in the second group, which reads input channels 2 and 3 only.
        self.assertEqual(sorted(entry["channel"] for entry in corner["channels"]), [2, 3])
        self.assertEqual(corner["channelRest"], {"count": 0, "sum": 0.0})
        for term in corner["terms"]:
            if term["input"] is None:
                self.assertEqual((term["x"], term["term"]), (0.0, 0.0))
            else:
                self.assertIn(term["input"][0], (2, 3))
                self.assertTrue(0 <= term["input"][1] <= 1 and 0 <= term["input"][2] <= 1)
            self.assertEqual(term["parameter"][0], 5)
        self.assertAlmostEqual(sum(entry["sum"] for entry in corner["channels"]) + corner["bias"], corner["recomputed"], places=9)
        self.assertAddsUp(corner)
        _, centre = self.explain(graph, "n1", [1, 3, 2])
        self.assertEqual((centre["padded"], sorted(entry["channel"] for entry in centre["channels"])), (0, [0, 1]))
        self.assertAddsUp(centre)

    def test_a_strided_convolution_window_starts_at_stride_times_the_position(self):
        graph = chain(("Input", {"shape": [1, 1, 7, 7]}), ("Conv2d", {"out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1}), ("Output", {}))
        # Output (3, 3) reads rows and columns 5..7; the input ends at 6.
        _, explanation = self.explain(graph, "n1", [0, 3, 3])
        self.assertEqual((explanation["count"], explanation["padded"]), (9, 5))
        read = {tuple(term["input"]) for term in explanation["terms"] if term["input"] is not None}
        self.assertEqual(read, {(0, row, column) for row in (5, 6) for column in (5, 6)})
        self.assertAddsUp(explanation)

    def test_max_pooling_is_the_largest_value_inside_its_window(self):
        graph = chain(("Input", {"shape": [1, 2, 5, 5]}), ("MaxPool2d", {"kernel_size": 3, "stride": 2, "padding": 1}), ("Output", {}))
        _, explanation = self.explain(graph, "n1", [1, 0, 0], inputs={"n0": [float(value) for value in range(50)]})
        # Rows and columns -1..1: four positions inside; the padding is -inf and never competes.
        self.assertEqual((explanation["rule"], explanation["count"], explanation["padded"], len(explanation["terms"])), ("max", 4, 5, 4))
        self.assertEqual(explanation["terms"][0], {"input": [1, 1, 1], "x": 31.0, "parameter": None, "w": None, "term": 31.0})
        self.assertEqual((explanation["value"], explanation["recomputed"]), (31.0, 31.0))
        self.assertEqual((explanation["bias"], explanation["divisor"], explanation["rest"]), (None, None, {"count": 0, "sum": None}))

    def test_average_pooling_divides_by_the_window_pytorch_counts(self):
        graph = chain(("Input", {"shape": [1, 1, 4, 4]}), ("AvgPool2d", {"kernel_size": 3, "stride": 1, "padding": 1}), ("Output", {}))
        _, explanation = self.explain(graph, "n1", [0, 0, 0], inputs={"n0": [float(value) for value in range(16)]})
        # count_include_pad: the padded zeros count too, so the corner divides by all nine.
        self.assertEqual((explanation["rule"], explanation["divisor"], explanation["count"], explanation["padded"]), ("mean", 9, 9, 5))
        self.assertAlmostEqual(explanation["value"], (0 + 1 + 4 + 5) / 9, places=6)
        self.assertAddsUp(explanation)

    def test_adaptive_pooling_uses_pytorch_s_floor_and_ceil_windows(self):
        sample = [float(value % 11) - 5 for value in range(3 * 5 * 7)]
        for op, rule in (("AdaptiveAvgPool2d", "mean"), ("AdaptiveMaxPool2d", "max")):
            graph = chain(("Input", {"shape": [1, 3, 5, 7]}), (op, {"output_size": 2}), ("Output", {}))
            with self.subTest(op=op):
                # Rows floor(1 * 5 / 2) = 2 to ceil(2 * 5 / 2) = 5, columns 3 to 7.
                _, explanation = self.explain(graph, "n1", [2, 1, 1], inputs={"n0": sample})
                self.assertEqual((explanation["rule"], explanation["count"], explanation["padded"]), (rule, 12, 0))
                self.assertEqual(explanation["divisor"], 12 if rule == "mean" else None)
                self.assertTrue(all(term["input"][0] == 2 and 2 <= term["input"][1] < 5 and 3 <= term["input"][2] < 7 for term in explanation["terms"]))
                self.assertAddsUp(explanation)

    def test_a_shared_or_folded_layer_explains_the_call_that_produced_its_value(self):
        graph = chain(("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 4}), ("Tanh", {}), ("Linear", {"out_features": 4}), ("Output", {}))
        graph["nodes"][3]["shares"] = "n1"
        _, shared = self.explain(graph, "n3", [1])
        _, own = self.explain(graph, "n1", [1])
        # One weight row, two calls: the sharing layer multiplies it with the Tanh outputs.
        self.assertEqual((shared["owner"], shared["inputNode"], own["inputNode"]), ("n1", "n2", "n0"))
        self.assertNotIn("owner", own)
        self.assertEqual(sorted(term["w"] for term in shared["terms"]), sorted(term["w"] for term in own["terms"]))
        self.assertNotEqual(sorted(term["x"] for term in shared["terms"]), sorted(term["x"] for term in own["terms"]))
        self.assertAddsUp(shared)
        self.assertAddsUp(own)
        folded = chain(("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 4}), ("Output", {}))
        folded["nodes"][1]["repeat"] = 3
        _, last = self.explain(folded, "n1", [0])
        self.assertEqual((last["instance"], last["instances"]), (2, 3))
        self.assertAddsUp(last)

    def test_only_a_selected_weighted_sum_or_pooling_value_inside_the_output_is_explained(self):
        graph = chain(("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 3}), ("ReLU", {}), ("Output", {}))
        for node, index, message in (("n2", [0], "cannot be explained"), ("n1", [3], "inside the output shape"), ("n1", [0, 0], "inside the output shape")):
            with self.subTest(node=node, index=index), self.assertRaisesRegex(InferenceError, message):
                self.explain(graph, node, index)
        with self.assertRaisesRegex(InferenceError, "selected nodeIds"):
            run_inference(TensorInferenceRequest(graph=graph, nodeIds=["n2"], explain={"nodeId": "n1", "index": [0]}))
        with self.assertRaises(ValueError):
            TensorInferenceRequest(graph=graph, allNodes=True, explain={"nodeId": "n1", "index": [0]})
        with patch("backend.inference.MAX_EXPLAINED_TERMS", 3), self.assertRaisesRegex(InferenceError, "4 terms"):
            self.explain(graph, "n1", [0])
        # The float64 scratch joins the inspection budget, checked before the model is built.
        with patch("backend.inference.MAX_CAPTURE_BYTES", 100), patch("backend.training.build_model") as build, self.assertRaisesRegex(InferenceError, "Value explanation scratch exceeds 64 MB"):
            self.explain(graph, "n1", [0])
        build.assert_not_called()
        self.assertIsNone(run_inference(TensorInferenceRequest(graph=graph, nodeIds=["n1"]))["explanation"])


if __name__ == "__main__":
    unittest.main()
