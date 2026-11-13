"""Focused inference contract and runtime tests; PyTorch tests skip when unavailable."""
import copy
import unittest
from unittest.mock import patch

from .inference import InferenceError, TensorInferenceRequest, _describe, run_inference
from .training import build_model

try:
    import torch
except ImportError:  # pragma: no cover - deployment without training extra
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class InferenceRuntimeTests(unittest.TestCase):
    @staticmethod
    def mlp_graph(batch=7):
        specs = [("Input", {"shape": [batch, 4]}), ("Linear", {"out_features": 3}), ("ReLU", {}), ("Output", {})]
        nodes = [{"id": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
        return {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(3)]}

    @staticmethod
    def embedding_graph():
        specs = [("Input", {"shape": [4, 3]}), ("Embedding", {"num_embeddings": 5, "embedding_dim": 4}), ("Flatten", {}), ("Linear", {"out_features": 2}), ("Output", {})]
        nodes = [{"id": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
        return {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(4)]}

    def request(self, graph, nodes=None, **kwargs):
        return TensorInferenceRequest(graph=graph, nodeIds=nodes or ["n0", "n1", "n3"], **kwargs)

    @staticmethod
    def deep_chain(layers=20, size=8):
        """A long chain whose total activation far exceeds any single layer's."""
        nodes = [{"id": "n0", "op": "Input", "params": {"shape": [1, size, size]}}]
        edges = []
        for index in range(layers):
            nodes.append({"id": f"n{index + 1}", "op": "ReLU", "params": {}})
            edges.append({"source": f"n{index}", "target": f"n{index + 1}"})
        nodes.append({"id": f"n{layers + 1}", "op": "Output", "params": {}})
        edges.append({"source": f"n{layers}", "target": f"n{layers + 1}"})
        return {"version": 1, "name": "deep", "nodes": nodes, "edges": edges}

    def test_shallow_inspection_is_budgeted_by_its_ancestor_cone_not_the_whole_model(self):
        # 20 层每层 256 B 合计 5120 B，但浅层的祖先锥只有 512 B。闸门若按整模型总和计，
        # 一个只想看第一层的请求会被无关的深层拒绝——这正是大模型单样本检查的瓶颈。
        graph = self.deep_chain(layers=20, size=8)
        limit = 1024
        with patch("backend.inference.MAX_CAPTURE_BYTES", limit):
            shallow = run_inference(self.request(graph, nodes=["n1"]))
            self.assertEqual(shallow["tensors"][0]["nodeId"], "n1")
            # 末层的祖先锥覆盖整条链，仍然必须被拒绝。
            with self.assertRaisesRegex(ValueError, "64 MB"):
                run_inference(self.request(graph, nodes=["n20"]))

    def test_all_nodes_observes_every_layer_with_the_numbers_a_selection_gets(self):
        graph = self.mlp_graph()
        everything = run_inference(TensorInferenceRequest(graph=graph, allNodes=True, seed=5))
        self.assertEqual([tensor["nodeId"] for tensor in everything["tensors"]], ["n0", "n1", "n2", "n3"])
        self.assertEqual(everything["tensors"][1], run_inference(self.request(graph, ["n1"], seed=5))["tensors"][0])
        # Every layer and nothing else, or named layers: never both, never neither.
        for payload in ({"allNodes": True, "nodeIds": ["n1"]}, {"allNodes": True, "attention": {"n1": {}}}, {}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                TensorInferenceRequest(graph=graph, **payload)
        with patch("backend.inference.MAX_ALL_NODES", 3), self.assertRaisesRegex(InferenceError, "at most 3"):
            run_inference(TensorInferenceRequest(graph=graph, allNodes=True))

    def test_deterministic_seed_and_batch_reduction(self):
        graph = self.mlp_graph(); original = copy.deepcopy(graph)
        first = run_inference(self.request(graph))
        second = run_inference(self.request(graph))
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["tensors"], second["tensors"])
        self.assertEqual(first["tensors"][0]["shape"], [1, 4])
        self.assertEqual(graph, original)

    def test_seed_changes_random_initialized_result(self):
        graph = self.mlp_graph()
        first = run_inference(self.request(graph, seed=7))
        second = run_inference(self.request(graph, seed=8))
        self.assertNotEqual(first["tensors"][-1]["stats"], second["tensors"][-1]["stats"])

    def test_provided_input_and_slice_are_reported(self):
        graph = self.mlp_graph()
        result = run_inference(self.request(graph, ["n0"], inputs={"n0": [1, 2, 3, 4]}, slices={"n0": []}))
        tensor = result["tensors"][0]
        self.assertEqual(result["inputSource"], "provided")
        self.assertEqual(tensor["shape"], [1, 4])
        self.assertEqual(tensor["slice"]["values"], [[1.0, 2.0, 3.0, 4.0]])

    def test_stats_histogram_and_nonfinite_values(self):
        value = torch.tensor([float("nan"), float("inf"), 2.0, 2.0])
        result = _describe("x", value, [], torch)
        self.assertEqual(result["finiteCount"], 2)
        self.assertEqual(result["nonFiniteCount"], 2)
        self.assertEqual(result["stats"]["mean"], 2.0)
        self.assertEqual(sum(result["histogram"]["counts"]), 2)
        constant = _describe("x", torch.full((4,), 2.0), [], torch)
        self.assertEqual(sum(constant["histogram"]["counts"]), 4)
        self.assertEqual(constant["histogram"]["counts"][6], 4)

    def test_observer_captures_input_and_actual_ops(self):
        graph = self.mlp_graph(); model, _ = build_model(graph); seen = {}
        model(torch.randn(1, 4), observer=lambda key, value: seen.setdefault(key, value))
        self.assertEqual(set(seen), {"n0", "n1", "n2", "n3"})

    def test_observer_does_not_change_normal_output(self):
        graph = self.mlp_graph(); model, _ = build_model(graph); x = torch.randn(1, 4)
        expected = model(x)
        observed = model(x, observer=lambda *_: None)
        torch.testing.assert_close(expected, observed)

    def test_embedding_accepts_integer_ids_and_rejects_float_ids(self):
        graph = self.embedding_graph()
        result = run_inference(self.request(graph, ["n0", "n1"], inputs={"n0": [0, 1, 2]}))
        self.assertEqual(result["tensors"][0]["dtype"], "int64")
        with self.assertRaises(ValueError):
            run_inference(self.request(graph, ["n0"], inputs={"n0": [0, 1.5, 2]}))
        with self.assertRaises(ValueError):
            run_inference(self.request(graph, ["n0"], inputs={"n0": [0, 1, 5]}))

    def test_global_rng_is_unchanged(self):
        graph = self.mlp_graph(); torch.manual_seed(101); expected = torch.rand(4)
        torch.manual_seed(101); run_inference(self.request(graph)); actual = torch.rand(4)
        torch.testing.assert_close(expected, actual)

    def test_limits_are_checked_before_model_construction(self):
        graph = self.mlp_graph(); graph["nodes"][0]["params"]["shape"] = [7, 65536]
        graph["nodes"][1]["params"]["out_features"] = 128
        with patch("backend.training.build_model", side_effect=AssertionError("allocated too early")):
            with self.assertRaisesRegex(ValueError, "parameter"):
                run_inference(self.request(graph, ["n0"]))

    def test_functional_add_slice_select_observed(self):
        nodes = [{"id": "input", "op": "Input", "params": {"shape": [3, 20, 20]}}, {"id": "identity", "op": "Identity", "params": {}}, {"id": "add", "op": "Add", "params": {}}, {"id": "slice", "op": "Slice", "params": {"dim": 2, "start": 0, "end": 3}}, {"id": "select", "op": "Select", "params": {"dim": 2, "index": 1}}, {"id": "output", "op": "Output", "params": {}}]
        edges = [{"source": "input", "target": "identity"}, {"source": "input", "target": "add"}, {"source": "identity", "target": "add"}, {"source": "add", "target": "slice"}, {"source": "slice", "target": "select"}, {"source": "select", "target": "output"}]
        graph = {"version": 1, "nodes": nodes, "edges": edges}
        values = [float(i) for i in range(400)]
        result = run_inference(self.request(graph, ["add", "slice", "select"], inputs={"input": values}, slices={"add": [0], "slice": [0], "select": []}))
        self.assertEqual([tensor["nodeId"] for tensor in result["tensors"]], ["add", "slice", "select"])
        self.assertEqual(result["tensors"][0]["slice"]["shape"], [20, 20])
        self.assertTrue(result["tensors"][0]["slice"]["truncated"])

    def test_endpoint_lock_busy_and_release(self):
        from .app import infer, training_lock
        graph = self.mlp_graph(batch=1)
        payload = self.request(graph)
        self.assertFalse(training_lock.locked())
        with training_lock:
            with self.assertRaises(Exception) as caught:
                infer(payload)
            self.assertEqual(getattr(caught.exception, "status_code", None), 409)
        infer(payload)
        self.assertFalse(training_lock.locked())
        invalid = self.request(graph, ["missing"])
        with self.assertRaises(Exception) as caught: infer(invalid)
        self.assertEqual(getattr(caught.exception, "status_code", None), 422)
        self.assertFalse(training_lock.locked())

    def test_cross_attention_multiple_inputs_and_projection_edges(self):
        from .test_training import cross_graph
        graph = cross_graph()
        graph["nodes"].append({"id": "projected", "op": "Identity", "params": {}})
        graph["edges"].append({"source": "layer_1", "sourcePort": "b0:q0", "target": "projected"})
        graph["edges"].append({"source": "context", "target": "layer_1", "targetPort": "b0:k0"})
        result = run_inference(self.request(graph, ["layer_1", "projected", "layer_4"]))
        self.assertEqual([tensor["shape"] for tensor in result["tensors"]], [[1, 4, 8], [1, 4, 4], [1, 2]])
        with self.assertRaisesRegex(ValueError, "exactly every Input"):
            run_inference(self.request(graph, ["layer_1"], inputs={"layer_0": [0] * 32}))

    def test_shared_embedding_input_uses_smallest_vocabulary(self):
        graph = self.embedding_graph()
        graph["nodes"].extend([
            {"id": "other", "op": "Embedding", "params": {"num_embeddings": 3, "embedding_dim": 4}},
            {"id": "sum", "op": "Add", "params": {}},
        ])
        graph["edges"] = [edge for edge in graph["edges"] if edge["target"] != "n2"] + [
            {"source": "n0", "target": "other"}, {"source": "n1", "target": "sum"},
            {"source": "other", "target": "sum"}, {"source": "sum", "target": "n2"},
        ]
        result = run_inference(self.request(graph, ["n0", "other"]))
        self.assertLess(result["tensors"][0]["stats"]["max"], 3)

    def test_all_nonfinite_and_extreme_float32_statistics_are_json_safe(self):
        import json
        for value in (torch.tensor([float("nan"), float("inf"), -float("inf")]), torch.tensor([-3e38, 0.0, 3e38])):
            tensor = _describe("x", value, [], torch)
            json.dumps(tensor, allow_nan=False)
            self.assertEqual(sum(tensor["histogram"]["counts"]), tensor["finiteCount"])
        self.assertEqual(_describe("x", torch.tensor([float("inf")]), [], torch)["stats"]["mean"], None)

    def test_channel_slice_crops_actual_selected_values(self):
        value = torch.arange(800, dtype=torch.float32).reshape(1, 2, 20, 20)
        result = _describe("x", value, [0, 1], torch)
        self.assertEqual(result["slice"]["values"][0][:3], [400.0, 401.0, 402.0])
        self.assertEqual((result["slice"]["rows"], result["slice"]["columns"]), (16, 16))
        self.assertTrue(result["slice"]["truncated"])

    def test_channel_page_shows_the_same_window_for_the_channels_around_the_selected_one(self):
        # 40 channels page in sixteens, so channel 20 belongs to the page 16..31. Each map has to
        # be the very window the main slice would show for that channel.
        value = torch.arange(40 * 20 * 20, dtype=torch.float32).reshape(1, 40, 20, 20)
        result = _describe("x", value, [0, 20], torch)
        page = result["channels"]
        self.assertEqual((page["start"], page["count"], page["total"], page["selected"]), (16, 16, 40, 20))
        self.assertEqual((page["shape"], page["rows"], page["columns"], page["truncated"]), ([20, 20], 16, 16, True))
        self.assertEqual(len(page["maps"]), 16)
        self.assertEqual(page["maps"][20 - 16], result["slice"]["values"])
        for offset in (0, 7, 15):
            self.assertEqual(page["maps"][offset], _describe("x", value, [0, 16 + offset], torch)["slice"]["values"])

    def test_channel_page_statistics_cover_the_whole_channel_not_the_window(self):
        # Channel 1 is zero everywhere inside the 16 x 16 window and active only outside it. A
        # window statistic would call it dead; the channel statistic must not.
        value = torch.zeros(1, 3, 20, 20)
        value[0, 0] = -2.0
        value[0, 1, 19, 19] = 5.0
        value[0, 2, 0, 0] = float("nan"); value[0, 2, 1, 1] = 3.0
        page = _describe("x", value, [0, 0], torch)["channels"]
        self.assertEqual((page["start"], page["count"], page["total"]), (0, 3, 3))
        self.assertEqual(page["stats"][0], {"min": -2.0, "max": -2.0, "mean": -2.0})
        self.assertTrue(all(item == 0.0 for row in page["maps"][1] for item in row))
        self.assertEqual((page["stats"][1]["min"], page["stats"][1]["max"]), (0.0, 5.0))
        self.assertAlmostEqual(page["stats"][1]["mean"], 5.0 / 400)
        # A non-finite element is reported as such in the map and left out of the statistics.
        self.assertIsNone(page["maps"][2][0][0])
        self.assertEqual(page["stats"][2]["max"], 3.0)
        self.assertAlmostEqual(page["stats"][2]["mean"], 3.0 / 399)
        empty = torch.full((1, 1, 2, 2), float("inf"))
        self.assertEqual(_describe("x", empty, [0, 0], torch)["channels"]["stats"], [{"min": None, "max": None, "mean": None}])

    def test_channel_page_ends_at_the_last_channel_and_follows_the_depth_index(self):
        import json
        value = torch.arange(40 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 40, 3, 4, 5)
        result = _describe("x", value, [0, 35, 2], torch)
        page = result["channels"]
        # The last page is short rather than padded.
        self.assertEqual((page["start"], page["count"], page["selected"]), (32, 8, 35))
        self.assertEqual((page["shape"], page["rows"], page["columns"], page["truncated"]), ([4, 5], 4, 5, False))
        self.assertEqual(page["maps"][35 - 32], value[0, 35, 2].tolist())
        self.assertEqual(page["maps"][0], value[0, 32, 2].tolist())
        # Statistics still describe the whole channel volume, across every depth.
        self.assertEqual(page["stats"][0]["min"], float(value[0, 32].min()))
        self.assertEqual(page["stats"][0]["max"], float(value[0, 32].max()))
        json.dumps(result, allow_nan=False)

    def test_only_channel_first_spatial_tensors_get_a_channel_page(self):
        # A sequence has no channel axis to page through, and neither does a feature vector.
        self.assertIsNone(_describe("x", torch.randn(1, 6, 8), [0], torch)["channels"])
        self.assertIsNone(_describe("x", torch.randn(1, 8), [], torch)["channels"])
        self.assertIsNone(_describe("x", torch.randn(4), [], torch)["channels"])
        result = run_inference(self.request(self.mlp_graph(), ["n1"]))
        self.assertIsNone(result["tensors"][0]["channels"])

    def test_inference_reports_the_channel_page_of_the_requested_slice(self):
        specs = [("Input", {"shape": [3, 2, 6, 6]}), ("Conv2d", {"out_channels": 20, "kernel_size": 3, "stride": 1, "padding": 1}), ("Flatten", {}), ("Linear", {"out_features": 2}), ("Output", {})]
        nodes = [{"id": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
        graph = {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(4)]}
        result = run_inference(self.request(graph, ["n1"], slices={"n1": [0, 17]}))
        tensor = result["tensors"][0]
        self.assertEqual(tensor["shape"], [1, 20, 6, 6])
        page = tensor["channels"]
        self.assertEqual((page["start"], page["count"], page["total"], page["selected"]), (16, 4, 20, 17))
        self.assertEqual(page["maps"][1], tensor["slice"]["values"])
        self.assertFalse(page["truncated"])

    def test_single_sample_limits_do_not_apply_to_submitted_batch(self):
        graph = self.mlp_graph(batch=65536)
        self.assertEqual(run_inference(self.request(graph))["tensors"][0]["shape"], [1, 4])

    def test_attention_limit_counts_actual_override_lengths_before_allocation(self):
        from .test_training import cross_graph
        graph = cross_graph()
        graph["nodes"][0]["params"]["shape"] = [1, 256, 8]
        graph["nodes"][-1]["params"]["shape"] = [1, 1024, 8]
        graph["nodes"].append({"id": "long", "op": "Input", "params": {"shape": [1, 2048, 8]}})
        graph["edges"].extend({"source": "long", "target": "layer_1", "targetPort": f"b{branch}:{role}0"} for branch in range(2) for role in ("k", "v"))
        with patch("backend.training.build_model", side_effect=AssertionError("allocated too early")):
            with self.assertRaisesRegex(ValueError, "1M"):
                run_inference(self.request(graph, ["layer_1"]))


class InferenceRequestTests(unittest.TestCase):
    def graph(self):
        return {"version": 1, "nodes": [{"id": "input", "op": "Input", "params": {"shape": [1, 2]}}, {"id": "output", "op": "Output", "params": {}}], "edges": [{"source": "input", "target": "output"}]}

    def test_strict_request_fields_and_duplicates(self):
        graph = self.graph()
        with self.assertRaises(ValueError): TensorInferenceRequest(graph=graph, nodeIds=["input", "input"])
        with self.assertRaises(ValueError): TensorInferenceRequest(graph=graph, nodeIds=["input"], seed="42")
        with self.assertRaises(ValueError): TensorInferenceRequest(graph=graph, nodeIds=["input"], inputs={"input": [True, 0]})

    def test_slice_prefix_bound(self):
        with self.assertRaises(ValueError): TensorInferenceRequest(graph=self.graph(), nodeIds=["input"], slices={"input": [0, 0, 0, 0]})

    def test_fastapi_route_uses_json_body_model(self):
        from .app import app
        route = next(route for route in app.routes if getattr(route, "path", None) == "/api/infer")
        self.assertIs(route.dependant.body_params[0].type_, TensorInferenceRequest)


if __name__ == "__main__": unittest.main()
