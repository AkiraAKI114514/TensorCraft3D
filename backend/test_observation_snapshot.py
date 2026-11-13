"""Offline observation snapshot tests; PyTorch tests skip when unavailable."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .inference import InferenceError, TensorInferenceRequest, run_inference
from .observation_snapshot import FORMAT, VERSION, capture_snapshot, main, snapshot_json, write_snapshot
from .training import build_model

try:
    import torch
except ImportError:  # pragma: no cover - deployment without training extra
    torch = None


def chain(*specs, batch=4):
    """Input -> specs... -> Output, with the Input shape as the first spec."""
    ops = [("Input", {"shape": [batch, *specs[0]]}), *specs[1:], ("Output", {})]
    nodes = [{"id": f"n{i}", "name": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(ops)]
    return {"version": 1, "name": "chain", "nodes": nodes, "edges": [{"id": f"e{i}", "source": f"n{i}", "target": f"n{i + 1}"} for i in range(len(ops) - 1)]}


def cnn(batch=4):
    return chain([1, 6, 6], ("Conv2d", {"out_channels": 3, "kernel_size": 3, "stride": 1, "padding": 1}), ("ReLU", {}), ("Flatten", {}), ("Linear", {"out_features": 2}), batch=batch)


def attention(**params):
    return chain([5, 8], ("MultiHeadAttention", {"embed_dim": 8, "num_heads": 2, "kv_heads": 2, "branches": 1, "attention_type": "self", "dropout": 0, **params}), ("Flatten", {}), ("Linear", {"out_features": 2}))


def trained(graph, seed=3):
    """Stand-in for weights trained elsewhere: anything but the snapshot's own seeded init."""
    torch.manual_seed(seed)
    model, _ = build_model(graph)
    return model.eval()


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class ObservationSnapshotTests(unittest.TestCase):
    def test_external_weights_and_sample_reproduce_the_models_own_forward(self):
        # Batch 1, so the reference model below can run the sample itself.
        graph = cnn(batch=1); model = trained(graph)
        sample = torch.randn(1, 1, 6, 6)
        snapshot = capture_snapshot(graph, model.state_dict(), {"n0": sample}, label="digit 7")
        self.assertEqual((snapshot["format"], snapshot["version"], snapshot["weights"], snapshot["inputSource"], snapshot["label"]), (FORMAT, VERSION, "external", "provided", "digit 7"))
        self.assertEqual(snapshot["graph"], graph)
        # Every node is observed, in execution order, so any layer can be inspected offline.
        self.assertEqual([tensor["nodeId"] for tensor in snapshot["tensors"]], [f"n{i}" for i in range(6)])
        by_id = {tensor["nodeId"]: tensor for tensor in snapshot["tensors"]}
        with torch.no_grad():
            conv = model.layers["n1"](sample); expected = model(sample)
        output = by_id["n5"]
        self.assertEqual(output["shape"], [1, 2])
        self.assertAlmostEqual(output["stats"]["min"], expected.min().item(), places=6)
        self.assertAlmostEqual(output["stats"]["max"], expected.max().item(), places=6)
        # Feature maps carry the channel page the inspector's grid shows.
        page = by_id["n1"]["channels"]
        self.assertEqual((page["start"], page["count"], page["total"]), (0, 3, 3))
        for channel in range(3):
            torch.testing.assert_close(torch.tensor(page["maps"][channel]), conv[0, channel])
        self.assertIsNone(by_id["n4"]["channels"])
        self.assertEqual(snapshot["attentions"], [])

    def test_holds_the_numbers_the_inspector_measures_for_the_same_weights_and_sample(self):
        graph = cnn()
        snapshot = capture_snapshot(graph, seed=11)
        self.assertEqual((snapshot["weights"], snapshot["inputSource"], snapshot["seed"]), ("random-initialized", "synthetic", 11))
        report = run_inference(TensorInferenceRequest(graph=graph, nodeIds=["n1", "n5"], seed=11))
        offline = {tensor["nodeId"]: tensor for tensor in snapshot["tensors"]}
        for tensor in report["tensors"]:
            self.assertEqual(offline[tensor["nodeId"]], tensor)

    def test_a_sample_keeps_or_drops_its_batch_axis_or_is_flat_but_is_never_reshaped(self):
        graph = cnn(); weights = trained(graph).state_dict(); sample = torch.randn(1, 1, 6, 6)
        reference = capture_snapshot(graph, weights, {"n0": sample})["tensors"][-1]
        for value in (sample[0], sample.reshape(-1), sample.tolist(), sample):
            self.assertEqual(capture_snapshot(graph, weights, value)["tensors"][-1]["stats"], reference["stats"])
        for value, message in ((sample.reshape(6, 6), r"must have shape"), ({"other": sample}, r"exactly the Input node IDs"), (torch.full((1, 1, 6, 6), float("nan")), r"non-finite")):
            with self.subTest(message=message), self.assertRaisesRegex(InferenceError, message):
                capture_snapshot(graph, weights, value)

    def test_weights_of_another_graph_are_refused_naming_what_differs(self):
        graph = cnn()
        other = trained(chain([1, 6, 6], ("Conv2d", {"out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1}), ("ReLU", {}), ("Flatten", {}), ("Linear", {"out_features": 2})))
        with self.assertRaisesRegex(InferenceError, r"do not match this graph"):
            capture_snapshot(graph, other.state_dict())
        state = trained(graph).state_dict(); state.pop("layers.n4.bias"); state["epoch"] = torch.tensor(3)
        with self.assertRaisesRegex(InferenceError, r"missing layers\.n4\.bias; unexpected epoch"):
            capture_snapshot(graph, state)
        with self.assertRaisesRegex(InferenceError, r"state dict"):
            capture_snapshot(graph, [1, 2, 3])

    def test_a_module_and_the_exported_layout_of_a_folded_block_both_load(self):
        block = {"id": "block", "name": "Block", "origin": "manual", "nodes": [{"id": "fc", "name": "fc", "op": "Linear", "params": {"out_features": 8}}, {"id": "act", "name": "act", "op": "Tanh", "params": {}}], "edges": [{"id": "s0", "source": "fc", "target": "act"}]}
        graph = chain([8], ("Group", {}), ("Linear", {"out_features": 2}))
        graph["nodes"][1].update(subgraph="block", repeat=2); graph["subgraphs"] = {"block": block}
        model = trained(graph); sample = torch.randn(1, 8)
        runtime = capture_snapshot(graph, model, sample)
        # Export writes a folded block as a module list of block classes; the runtime expands it.
        exported = {key.replace("layers.n1/0/fc.", "layers.n1.0.fc.").replace("layers.n1/1/fc.", "layers.n1.1.fc."): value for key, value in model.state_dict().items()}
        self.assertIn("layers.n1.1.fc.weight", exported)
        self.assertEqual(capture_snapshot(graph, exported, sample)["tensors"], runtime["tensors"])
        self.assertIn("n1/1/act", [tensor["nodeId"] for tensor in runtime["tensors"]])

    def test_attention_is_captured_at_a_default_window_chosen_or_left_out(self):
        graph = attention(causal=1)
        snapshot = capture_snapshot(graph)
        [captured] = snapshot["attentions"]
        self.assertEqual((captured["nodeId"], captured["branch"], captured["head"], captured["mask"]), ("n1", 0, 0, "causal"))
        self.assertEqual(captured["tensors"]["probabilities"]["slice"]["values"][0][1], 0.0)
        self.assertEqual(capture_snapshot(graph, attention={"n1": {"head": 1, "queryStart": 2}})["attentions"][0]["head"], 1)
        self.assertEqual(capture_snapshot(graph, attention={"n1": None})["attentions"], [])
        with self.assertRaisesRegex(InferenceError, r"requires an attention node"):
            capture_snapshot(graph, attention={"n2": {"head": 0}})

    def test_the_command_line_writes_a_file_and_reports_failures_without_one(self):
        graph = cnn(); model = trained(graph)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "project.json").write_text(json.dumps(graph), encoding="utf-8")
            torch.save(model.state_dict(), root / "weights.pt")
            (root / "sample.json").write_text(json.dumps({"n0": torch.randn(36).tolist()}), encoding="utf-8")
            out = root / "sample.observation.json"
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = main([str(root / "project.json"), "--weights", str(root / "weights.pt"), "--inputs", str(root / "sample.json"), "--out", str(out), "--label", "from the CLI"])
            self.assertEqual(code, 0, printed.getvalue())
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual((written["weights"], written["inputSource"], written["label"], len(written["tensors"])), ("external", "provided", "from the CLI", 6))
            torch.save({"n0": torch.randn(1, 1, 6, 6)}, root / "sample.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root / "project.json"), "--inputs", str(root / "sample.pt"), "--out", str(root / "random.json")]), 0)
            self.assertEqual(json.loads((root / "random.json").read_text(encoding="utf-8"))["weights"], "random-initialized")
            (root / "bad.json").write_text(json.dumps({"x": [0] * 36}), encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                self.assertEqual(main([str(root / "project.json"), "--inputs", str(root / "bad.json"), "--out", str(root / "never.json")]), 1)
            self.assertIn("Input node IDs", errors.getvalue())
            self.assertFalse((root / "never.json").exists())

    def test_a_snapshot_the_workbench_would_refuse_is_never_written(self):
        snapshot = capture_snapshot(cnn())
        self.assertNotIn("NaN", snapshot_json(snapshot))
        with tempfile.TemporaryDirectory() as folder, patch("backend.observation_snapshot.MAX_SNAPSHOT_BYTES", 1024):
            target = Path(folder) / "large.json"
            with self.assertRaisesRegex(InferenceError, r"more than the 1024 bytes"):
                write_snapshot(snapshot, target)
            self.assertFalse(target.exists())
        with self.assertRaisesRegex(InferenceError, r"label"):
            capture_snapshot(cnn(), label="x" * 121)


if __name__ == "__main__":
    unittest.main()
