import copy
import threading
import unittest
import torch
from .graph import analyze_graph
from .training import build_model, train


def constant_graph():
    specs = [("Input", {"shape": [2, 6, 4]}), ("ConstantAdd", {"shape": [1, 8, 4], "values": [v / 10 for v in range(32)], "sequence_dim": 1}), ("Slice", {"dim": 1, "start": 1, "end": "none", "step": 2}), ("Select", {"dim": 1, "index": -1}), ("Linear", {"out_features": 2}), ("Output", {})]
    nodes = [{"id": f"layer_{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
    return {"version": 1, "nodes": nodes, "edges": [{"source": f"layer_{i}", "target": f"layer_{i+1}"} for i in range(len(nodes) - 1)]}


class TensorOperationTests(unittest.TestCase):
    def test_constants_and_slices_preserve_values_gradients_and_dynamic_sizes(self):
        model, info = build_model(constant_graph())
        self.assertEqual(info["shapes"]["layer_2"], [2, 3, 4])
        self.assertEqual(info["shapes"]["layer_3"], [2, 4])
        self.assertEqual(info["totalParameters"], 10)
        self.assertIn("layers.layer_1.constant", model.state_dict())
        self.assertNotIn("layer_1", dict(model.named_parameters()))
        constant = torch.arange(32).float().reshape(1, 8, 4) / 10
        for batch, length in ((1, 4), (3, 6), (2, 8)):
            with self.subTest(batch=batch, length=length):
                x = torch.randn(batch, length, 4, requires_grad=True)
                expected = model.layers["layer_4"]((x + constant[:, :length])[:, 1::2][:, -1])
                actual = model(x)
                torch.testing.assert_close(actual, expected)
                a = torch.autograd.grad(actual.square().sum(), x)[0]
                b = torch.autograd.grad(expected.square().sum(), x)[0]
                torch.testing.assert_close(a, b)
        with self.assertRaisesRegex(ValueError, "capacity"):
            model(torch.randn(2, 9, 4))

    def test_negative_slices_match_pytorch(self):
        for start, end, step in ((-5, -1, 2), (-100, 100, 3), ("none", -2, 1), (2, "none", 2)):
            graph = constant_graph(); graph["nodes"][2]["params"].update(start=start, end=end, step=step)
            model, info = build_model(graph)
            x = torch.randn(2, 6, 4)
            s = slice(None if start == "none" else start, None if end == "none" else end, step)
            expected = model.layers["layer_4"]((x + model.layers["layer_1"].constant[:, :6])[:, s][:, -1])
            torch.testing.assert_close(model(x), expected)
            self.assertEqual(info["shapes"]["layer_2"][1], len(range(6)[s]))

    def test_rejects_invalid_buffers_and_batch_indexing_before_allocation(self):
        invalid = [(1, {"shape": [2, 4], "values": [0] * 8}), (1, {"shape": [1, 5, 4], "values": [0] * 20}), (1, {"values": [float("nan")] * 32}), (1, {"values": [1e39] * 32}), (1, {"sequence_dim": 0}), (1, {"shape": [1, 8, 3], "values": [0] * 24}), (2, {"dim": 0}), (2, {"step": 0}), (2, {"step": -1}), (2, {"start": 5, "end": 1}), (3, {"dim": 0}), (3, {"index": 100})]
        for node, params in invalid:
            with self.subTest(node=node, params=params):
                graph = constant_graph(); graph["nodes"][node]["params"].update(params)
                with self.assertRaises(ValueError): analyze_graph(graph)
        graph = constant_graph()
        graph["nodes"][1]["params"] = {"shape": [1, 16385, 4], "values": [0] * 65540, "sequence_dim": 1}
        with self.assertRaises(ValueError): analyze_graph(graph)
        graph = constant_graph()
        graph["nodes"][1]["params"] = {"shape": [1, 16384, 4], "values": [0] * 65536, "sequence_dim": 1}
        duplicate = copy.deepcopy(graph["nodes"][1]); duplicate["id"] = "constant2"
        graph["nodes"].insert(2, duplicate)
        graph["edges"][1] = {"source": "layer_1", "target": "constant2"}
        graph["edges"].append({"source": "constant2", "target": "layer_2"})
        with self.assertRaisesRegex(ValueError, "65536"): analyze_graph(graph)

    def test_fixed_scalar_buffer_addition(self):
        graph = constant_graph(); graph["nodes"][1]["params"] = {"shape": [], "values": [2]}
        model, _ = build_model(graph)
        x = torch.randn(3, 6, 4)
        torch.testing.assert_close(model(x), model.layers["layer_4"]((x + 2)[:, 1::2][:, -1]))

    def test_real_cpu_training_with_buffers_and_slices(self):
        messages = []
        config = {"epochs": 2, "learningRate": 0.01, "batchSize": 16, "samples": 64, "device": "cpu", "dataset": "synthetic", "validationFraction": 0.2, "earlyStopping": False, "patience": 5}
        train(constant_graph(), config, messages.append, threading.Event())
        metrics = [m["metric"] for m in messages if m["type"] == "metric"]
        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(m["source"] == "training" and m["layerGradients"]["layer_4"] > 0 for m in metrics))
        self.assertEqual(messages[-1], {"type": "done", "reason": "completed"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available in this PyTorch installation")
    def test_cuda_buffers_follow_model_device_and_backward(self):
        model, _ = build_model(constant_graph()); model.cuda()
        self.assertTrue(model.layers["layer_1"].constant.is_cuda)
        x = torch.randn(3, 6, 4, device="cuda", requires_grad=True)
        model(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__": unittest.main()
