"""Regression tests for retained training snapshots and trained inference."""
import copy
import json
import threading
import unittest
from unittest.mock import patch

from .inference import TensorInferenceRequest, run_inference
from .trained_models import MAX_SNAPSHOT_BYTES, TrainedModelStore, graph_fingerprint
from .training import build_model, train

try:
    import torch
except ImportError:  # pragma: no cover - deployment without training extra
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TrainedModelRuntimeTests(unittest.TestCase):
    @staticmethod
    def linear_graph(out_features=2, input_shape=None):
        input_shape = input_shape or [1, 4]
        specs = [
            ("Input", {"shape": list(input_shape)}),
            ("Linear", {"out_features": out_features}),
            ("Output", {}),
        ]
        nodes = [{"id": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
        return {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(2)]}

    @staticmethod
    def batch_norm_graph():
        specs = [
            ("Input", {"shape": [1, 2, 2]}),
            ("BatchNorm1d", {}),
            ("Flatten", {}),
            ("Linear", {"out_features": 2}),
            ("Output", {}),
        ]
        nodes = [{"id": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
        return {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(4)]}

    @staticmethod
    def dropout_graph():
        specs = [
            ("Input", {"shape": [1, 4]}),
            ("Dropout", {"p": 0.5}),
            ("Linear", {"out_features": 2}),
            ("Output", {}),
        ]
        nodes = [{"id": f"n{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
        return {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(3)]}

    @staticmethod
    def config(**overrides):
        config = {
            "epochs": 2,
            "learningRate": 0.01,
            "batchSize": 16,
            "samples": 64,
            "device": "cpu",
            "dataset": "synthetic",
            "validationFraction": 0.2,
            "earlyStopping": False,
            "patience": 5,
        }
        config.update(overrides)
        return config

    def retain_training(self, graph, store, **overrides):
        messages = []
        observed = {}

        def retain(actual_graph, model, preprocessing, provenance):
            observed["model"] = model
            observed["preprocessing"] = preprocessing
            observed["provenance"] = provenance
            return store.save(actual_graph, model, preprocessing, provenance)

        train(graph, self.config(**overrides), messages.append, threading.Event(), retain_model=retain)
        return messages, observed

    @staticmethod
    def inference(graph, model_id, values, node_ids=None, seed=42, device="cpu"):
        payload = {
            "graph": graph,
            "nodeIds": node_ids or ["n2"],
            "modelId": model_id,
            "seed": seed,
            "device": device,
            "inputs": values if isinstance(values, dict) else {"n0": values},
        }
        # Exercise the same JSON boundary as the HTTP endpoint.
        return run_inference(TensorInferenceRequest.model_validate(json.loads(json.dumps(payload))))

    def test_actual_training_retains_completed_final_state(self):
        graph = self.linear_graph()
        store = TrainedModelStore()
        messages, observed = self.retain_training(graph, store, epochs=2)

        self.assertEqual(messages[-1]["type"], "done")
        self.assertEqual(messages[-1]["reason"], "completed")
        metadata = messages[-1]["model"]
        self.assertEqual(metadata["reason"], "completed")
        self.assertEqual(metadata["epochsCompleted"], 2)
        self.assertEqual(metadata["weightsEpoch"], 2)
        snapshot = store.get(metadata["modelId"], graph)
        self.assertEqual(snapshot.metadata, metadata)
        for key, value in observed["model"].state_dict().items():
            torch.testing.assert_close(snapshot.state[key], value.detach().cpu())

    def test_early_stopping_retains_restored_best_state_and_epoch(self):
        graph = self.linear_graph()
        store = TrainedModelStore()
        messages, observed = self.retain_training(
            graph,
            store,
            epochs=10,
            learningRate=0,
            earlyStopping=True,
            patience=3,
        )

        done = messages[-1]
        self.assertEqual(done["reason"], "early_stopping")
        self.assertEqual(done["model"]["epochsCompleted"], 4)
        self.assertEqual(done["model"]["weightsEpoch"], 1)
        snapshot = store.get(done["model"]["modelId"], graph)
        for key, value in observed["model"].state_dict().items():
            torch.testing.assert_close(snapshot.state[key], value.detach().cpu())

    def test_early_stopping_restores_best_batch_norm_buffers(self):
        graph = self.batch_norm_graph()
        store = TrainedModelStore()
        tracked = {"epochs": {}}
        original_build = build_model

        def wrapped_build(actual_graph):
            model, info = original_build(actual_graph)
            tracked["model"] = model
            original_load = model.load_state_dict

            def tracked_load(state, *args, **kwargs):
                tracked["restored"] = {key: value.detach().cpu().clone() for key, value in state.items()}
                return original_load(state, *args, **kwargs)

            model.load_state_dict = tracked_load
            return model, info

        class ConstantLoss(torch.nn.Module):
            def forward(self, prediction, target):
                return prediction.sum() * 0 + 1.0

        messages = []
        observed = {}

        def retain(actual_graph, model, preprocessing, provenance):
            observed["state"] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            return store.save(actual_graph, model, preprocessing, provenance)

        def emit(message):
            messages.append(message)
            if message["type"] == "metric":
                tracked["epochs"][message["metric"]["epoch"]] = {
                    key: value.detach().cpu().clone() for key, value in tracked["model"].state_dict().items()
                }

        config = self.config(epochs=10, learningRate=0.01, earlyStopping=True, patience=3)
        with patch("backend.training.build_model", wrapped_build), patch("torch.nn.CrossEntropyLoss", ConstantLoss):
            train(graph, config, emit, threading.Event(), retain_model=retain)

        done = messages[-1]
        self.assertEqual(done["reason"], "early_stopping")
        self.assertEqual(done["model"]["epochsCompleted"], 4)
        self.assertEqual(done["model"]["weightsEpoch"], 1)
        self.assertIn("restored", tracked)
        self.assertEqual(set(tracked["epochs"]), {1, 2, 3, 4})
        self.assertIn("layers.n1.running_mean", tracked["restored"])
        self.assertIn("layers.n1.running_var", tracked["restored"])
        for key, value in tracked["epochs"][1].items():
            torch.testing.assert_close(tracked["restored"][key], value)
        self.assertFalse(torch.equal(tracked["epochs"][1]["layers.n1.running_mean"], tracked["epochs"][4]["layers.n1.running_mean"]))
        snapshot = store.get(done["model"]["modelId"], graph)
        for key, value in observed["state"].items():
            torch.testing.assert_close(snapshot.state[key], value)

    def test_stopped_and_failed_training_never_retains(self):
        graph = self.linear_graph()
        store = TrainedModelStore()
        calls = []

        def retain(*args):
            calls.append(args)
            return store.save(*args)

        stop = threading.Event()
        stop.set()
        messages = []
        train(graph, self.config(), messages.append, stop, retain_model=retain)
        self.assertEqual(messages, [{"type": "done", "reason": "stopped"}])
        self.assertEqual(calls, [])
        self.assertIsNone(store._latest)

        invalid_graph = self.linear_graph(out_features=1)
        with self.assertRaisesRegex(ValueError, "classification logits"):
            train(invalid_graph, self.config(), lambda message: None, threading.Event(), retain_model=retain)
        self.assertEqual(calls, [])
        self.assertIsNone(store._latest)

    def test_snapshot_is_singleton_cpu_clone_and_size_bounded(self):
        graph = self.linear_graph()
        store = TrainedModelStore()
        model, _ = build_model(graph)
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}
        first = store.save(graph, model, {}, {"reason": "completed"})
        self.assertIsNotNone(first)
        with torch.no_grad():
            for value in model.parameters():
                value.add_(10)
        snapshot = store.get(first["modelId"], graph)
        for key, value in original.items():
            torch.testing.assert_close(snapshot.state[key], value)
            self.assertEqual(snapshot.state[key].device.type, "cpu")
        state_bytes = sum(value.numel() * value.element_size() for value in snapshot.state.values())
        self.assertLessEqual(state_bytes, MAX_SNAPSHOT_BYTES)

        replacement = store.save(graph, build_model(graph)[0], {}, {"reason": "completed"})
        self.assertNotEqual(first["modelId"], replacement["modelId"])
        with self.assertRaisesRegex(ValueError, "snapshot is unavailable"):
            store.get(first["modelId"], graph)

        with patch("backend.trained_models.MAX_SNAPSHOT_BYTES", 1):
            self.assertIsNone(store.save(graph, build_model(graph)[0], {}, {"reason": "completed"}))
        self.assertIsNotNone(store.get(replacement["modelId"], graph))

    def test_snapshot_limit_includes_preprocessing_tensors(self):
        graph = self.linear_graph()
        model, _ = build_model(graph)
        preprocessing = {"n0": {"mean": torch.zeros(4), "scale": torch.ones(4)}}
        state_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
        preprocessing_bytes = sum(value.numel() * value.element_size() for pair in preprocessing.values() for value in pair.values())
        store = TrainedModelStore()
        with patch("backend.trained_models.MAX_SNAPSHOT_BYTES", state_bytes + preprocessing_bytes - 1):
            self.assertIsNone(store.save(graph, model, preprocessing, {"reason": "completed"}))
        with patch("backend.trained_models.MAX_SNAPSHOT_BYTES", state_bytes + preprocessing_bytes):
            self.assertIsNotNone(store.save(graph, model, preprocessing, {"reason": "completed"}))

    def test_graph_fingerprint_ignores_display_metadata_but_covers_structure(self):
        graph = self.linear_graph()
        for node in graph["nodes"]:
            node["label"] = f"label-{node['id']}"
            node["position"] = {"x": 100, "y": 200}
        for index, edge in enumerate(graph["edges"]):
            edge["id"] = f"edge-{index}"
            edge["label"] = "display-only"
            edge["position"] = {"x": index}
        baseline = graph_fingerprint(graph)
        display_variant = copy.deepcopy(graph)
        for node in display_variant["nodes"]:
            node["label"] = "renamed"
            node["position"] = {"x": -1, "y": -2}
        for edge in display_variant["edges"]:
            edge["id"] = "new-edge-id"
            edge["label"] = "renamed"
            edge["position"] = {"x": -1}
        self.assertEqual(baseline, graph_fingerprint(display_variant))

        mutations = []
        changed = copy.deepcopy(graph)
        changed["nodes"][0]["params"]["shape"] = [2, 4]
        mutations.append(changed)
        changed = copy.deepcopy(graph)
        changed["nodes"][1]["params"]["out_features"] = 3
        mutations.append(changed)
        changed = copy.deepcopy(graph)
        changed["edges"][0]["sourcePort"] = "b0:q0"
        mutations.append(changed)
        changed = copy.deepcopy(graph)
        changed["edges"] = list(reversed(changed["edges"]))
        mutations.append(changed)
        for variant in mutations:
            with self.subTest(variant=variant):
                self.assertNotEqual(baseline, graph_fingerprint(variant))

    def test_csv_normalization_is_training_split_only_and_replayed_for_raw_inference(self):
        graph = self.linear_graph()
        rows = [[float(index + offset) for offset in range(4)] + [index % 2] for index in range(16)]
        csv = "\n".join(",".join(str(value) for value in row) for row in rows)
        store = TrainedModelStore()
        messages, observed = self.retain_training(graph, store, dataset="csv", csv=csv)
        metadata = messages[-1]["model"]
        snapshot = store.get(metadata["modelId"], graph)
        raw_features = torch.tensor([row[:4] for row in rows], dtype=torch.float32)
        permutation = torch.randperm(16, generator=torch.Generator().manual_seed(7))
        fit = raw_features[permutation][3:]
        torch.testing.assert_close(snapshot.preprocessing["n0"]["mean"], fit.mean(0))
        torch.testing.assert_close(snapshot.preprocessing["n0"]["scale"], fit.std(0).clamp_min(1e-6))
        torch.testing.assert_close(observed["preprocessing"]["n0"]["mean"], snapshot.preprocessing["n0"]["mean"])

        with patch("backend.trained_models.trained_models", store):
            actual = self.inference(graph, metadata["modelId"], [20, 21, 22, 23], node_ids=["n0", "n2"])
        self.assertEqual(actual["weights"], "trained")
        self.assertEqual(actual["model"]["modelId"], metadata["modelId"])
        self.assertEqual(actual["inputTransform"], "csv-standardized")
        input_tensor = next(tensor for tensor in actual["tensors"] if tensor["nodeId"] == "n0")
        expected_input = (torch.tensor([[20, 21, 22, 23]], dtype=torch.float32) - snapshot.preprocessing["n0"]["mean"]) / snapshot.preprocessing["n0"]["scale"]
        torch.testing.assert_close(torch.tensor(input_tensor["slice"]["values"]), expected_input)
        expected_model = build_model(graph)[0]
        expected_model.load_state_dict(snapshot.state)
        expected_model.eval()
        expected_input = (torch.tensor([[20, 21, 22, 23]], dtype=torch.float32) - snapshot.preprocessing["n0"]["mean"]) / snapshot.preprocessing["n0"]["scale"]
        with torch.inference_mode():
            expected_output = expected_model(expected_input)
        self.assertEqual(actual["inputSource"], "provided")
        output_tensor = next(tensor for tensor in actual["tensors"] if tensor["nodeId"] == "n2")
        torch.testing.assert_close(torch.tensor(output_tensor["slice"]["values"]), expected_output)
    def test_batch_norm_buffers_and_eval_output_match_retained_state(self):
        graph = self.batch_norm_graph()
        store = TrainedModelStore()
        messages, _ = self.retain_training(graph, store, epochs=2)
        metadata = messages[-1]["model"]
        snapshot = store.get(metadata["modelId"], graph)
        self.assertIn("layers.n1.running_mean", snapshot.state)
        self.assertIn("layers.n1.running_var", snapshot.state)

        with patch("backend.trained_models.trained_models", store):
            actual = self.inference(graph, metadata["modelId"], [1, 2, 3, 4], node_ids=["n0", "n4"])
        expected_model = build_model(graph)[0]
        expected_model.load_state_dict(snapshot.state)
        expected_model.eval()
        with torch.inference_mode():
            expected_output = expected_model(torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32))
        torch.testing.assert_close(torch.tensor(actual["tensors"][-1]["slice"]["values"]), expected_output)

    def test_multi_input_csv_replays_separate_standardization(self):
        from .test_training import bilinear_graph

        graph = bilinear_graph(in2_features=3, second_shape=(4, 3))
        rows = []
        for index in range(16):
            rows.append([float(index + offset) for offset in range(8)] + [float(index + offset) for offset in range(3)] + [index % 2])
        csv = "\n".join(",".join(str(value) for value in row) for row in rows)
        store = TrainedModelStore()
        messages, _ = self.retain_training(graph, store, dataset="csv", csv=csv, epochs=2)
        metadata = messages[-1]["model"]
        snapshot = store.get(metadata["modelId"], graph)
        self.assertEqual(set(snapshot.preprocessing), {"layer_0", "second"})
        raw_first = torch.tensor([row[:8] for row in rows], dtype=torch.float32).reshape(16, 8)
        raw_second = torch.tensor([row[8:11] for row in rows], dtype=torch.float32).reshape(16, 3)
        permutation = torch.randperm(16, generator=torch.Generator().manual_seed(7))
        fit_first, fit_second = raw_first[permutation][3:], raw_second[permutation][3:]
        for key, fit in (("layer_0", fit_first), ("second", fit_second)):
            torch.testing.assert_close(snapshot.preprocessing[key]["mean"], fit.mean(0))
            torch.testing.assert_close(snapshot.preprocessing[key]["scale"], fit.std(0).clamp_min(1e-6))

        values = {"layer_0": rows[15][:8], "second": rows[15][8:11]}
        with patch("backend.trained_models.trained_models", store):
            actual = self.inference(graph, metadata["modelId"], values, node_ids=["layer_0", "second", "layer_2"])
        self.assertEqual(actual["inputSource"], "provided")
        self.assertEqual(actual["weights"], "trained")
        self.assertEqual(actual["inputTransform"], "csv-standardized")
        transformed = {}
        for key in values:
            transformed[key] = (torch.tensor([values[key]], dtype=torch.float32) - snapshot.preprocessing[key]["mean"]) / snapshot.preprocessing[key]["scale"]
            observed = next(tensor for tensor in actual["tensors"] if tensor["nodeId"] == key)
            torch.testing.assert_close(torch.tensor(observed["slice"]["values"]), transformed[key])
        expected_model = build_model(graph)[0]
        expected_model.load_state_dict(snapshot.state)
        expected_model.eval()
        with torch.inference_mode():
            expected_output = expected_model({key: value for key, value in transformed.items()})
        output = next(tensor for tensor in actual["tensors"] if tensor["nodeId"] == "layer_2")
        torch.testing.assert_close(torch.tensor(output["slice"]["values"]), expected_output)

    def test_mixed_embedding_csv_keeps_token_input_unnormalized(self):
        nodes = [
            {"id": "tokens", "op": "Input", "params": {"shape": [1, 3]}},
            {"id": "embedding", "op": "Embedding", "params": {"num_embeddings": 5, "embedding_dim": 4}},
            {"id": "features", "op": "Input", "params": {"shape": [1, 3, 4]}},
            {"id": "projection", "op": "Linear", "params": {"out_features": 4}},
            {"id": "add", "op": "Add", "params": {}},
            {"id": "flatten", "op": "Flatten", "params": {}},
            {"id": "logits", "op": "Linear", "params": {"out_features": 2}},
            {"id": "output", "op": "Output", "params": {}},
        ]
        edges = [
            {"source": "tokens", "target": "embedding"},
            {"source": "features", "target": "projection"},
            {"source": "embedding", "target": "add"},
            {"source": "projection", "target": "add"},
            {"source": "add", "target": "flatten"},
            {"source": "flatten", "target": "logits"},
            {"source": "logits", "target": "output"},
        ]
        graph = {"version": 1, "nodes": nodes, "edges": edges}
        rows = []
        for index in range(16):
            rows.append([float(index % 5), float((index + 1) % 5), float((index + 2) % 5)] + [float(index + offset) for offset in range(12)] + [index % 2])
        csv = "\n".join(",".join(str(value) for value in row) for row in rows)
        store = TrainedModelStore()
        messages, _ = self.retain_training(graph, store, dataset="csv", csv=csv, epochs=1)
        metadata = messages[-1]["model"]
        snapshot = store.get(metadata["modelId"], graph)
        self.assertNotIn("tokens", snapshot.preprocessing)
        self.assertIn("features", snapshot.preprocessing)
        values = {"tokens": rows[15][:3], "features": rows[15][3:15]}
        with patch("backend.trained_models.trained_models", store):
            actual = self.inference(graph, metadata["modelId"], values, node_ids=["tokens", "features", "output"])
        token_tensor = next(tensor for tensor in actual["tensors"] if tensor["nodeId"] == "tokens")
        self.assertEqual(token_tensor["dtype"], "int64")
        self.assertEqual(token_tensor["slice"]["values"], [values["tokens"]])
        feature_tensor = next(tensor for tensor in actual["tensors"] if tensor["nodeId"] == "features")
        expected_features = (torch.tensor([values["features"]], dtype=torch.float32).reshape(1, 3, 4) - snapshot.preprocessing["features"]["mean"]) / snapshot.preprocessing["features"]["scale"]
        torch.testing.assert_close(torch.tensor(feature_tensor["slice"]["values"]), expected_features.reshape(3, 4))

    def test_provided_inputs_are_seed_independent_with_retained_eval_weights(self):
        graph = self.dropout_graph()
        store = TrainedModelStore()
        messages, _ = self.retain_training(graph, store, epochs=2)
        model_id = messages[-1]["model"]["modelId"]
        with patch("backend.trained_models.trained_models", store):
            first = self.inference(graph, model_id, [0.5, -1, 2, 3], seed=7)
            second = self.inference(graph, model_id, [0.5, -1, 2, 3], seed=999)
        self.assertEqual(first["inputSource"], "provided")
        self.assertEqual(first["tensors"], second["tensors"])

    def test_expired_model_id_is_strict_and_graph_mismatch_fails_before_build(self):
        graph = self.linear_graph()
        store = TrainedModelStore()
        first_model, _ = build_model(graph)
        first = store.save(graph, first_model, {}, {"reason": "completed"})
        second_model, _ = build_model(graph)
        second = store.save(graph, second_model, {}, {"reason": "completed"})
        self.assertNotEqual(first["modelId"], second["modelId"])
        with patch("backend.trained_models.trained_models", store), patch("backend.training.build_model", side_effect=AssertionError("build should not run")):
            with self.assertRaisesRegex(ValueError, "snapshot is unavailable"):
                self.inference(graph, first["modelId"], [1, 2, 3, 4])

            mismatch = copy.deepcopy(graph)
            mismatch["nodes"][1]["params"]["out_features"] = 3
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.inference(mismatch, second["modelId"], [1, 2, 3, 4], node_ids=["n2"])

        with patch("backend.trained_models.trained_models", store):
            with self.assertRaisesRegex(ValueError, "snapshot is unavailable"):
                self.inference(graph, "expired", [1, 2, 3, 4])
            random = self.inference(graph, None, [1, 2, 3, 4], seed=11)
        self.assertEqual(random["weights"], "random-initialized")
        self.assertIsNone(random["model"])

    def test_lock_releases_when_model_snapshot_is_missing(self):
        from .app import infer, training_lock

        graph = self.linear_graph()
        request = TensorInferenceRequest(graph=graph, nodeIds=["n2"], modelId="expired", inputs={"n0": [1, 2, 3, 4]})
        self.assertFalse(training_lock.locked())
        with self.assertRaises(Exception) as caught:
            with patch("backend.trained_models.trained_models", TrainedModelStore()):
                infer(request)
        self.assertEqual(getattr(caught.exception, "status_code", None), 422)
        self.assertFalse(training_lock.locked())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA not available in this PyTorch installation")
    def test_cpu_snapshot_can_run_inference_on_cuda(self):
        graph = self.linear_graph()
        store = TrainedModelStore()
        messages, _ = self.retain_training(graph, store, epochs=2)
        model_id = messages[-1]["model"]["modelId"]
        with patch("backend.trained_models.trained_models", store):
            cpu = self.inference(graph, model_id, [0.5, -1, 2, 3], device="cpu")
            cuda = self.inference(graph, model_id, [0.5, -1, 2, 3], device="cuda")
        self.assertEqual(cuda["device"], "CUDA")
        torch.testing.assert_close(torch.tensor(cuda["tensors"][0]["slice"]["values"]), torch.tensor(cpu["tensors"][0]["slice"]["values"]), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
