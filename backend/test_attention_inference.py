"""Focused Attention Inspector contract tests.

The tests are skipped when the optional PyTorch training extra is unavailable.
"""
import json
import unittest
from unittest.mock import patch

from .inference import TensorInferenceRequest, run_inference
from .training import build_model

try:
    import torch
except ImportError:  # pragma: no cover - deployment without training extra
    torch = None


def attention_graph(*, kind="self", heads=4, kv_heads=None, branches=1, query=5, key=7, transformer=False, norm_first=True):
    kv_heads = heads if kv_heads is None else kv_heads
    op = "Transformer" if transformer else "MultiHeadAttention"
    params = {
        "embed_dim": 8,
        "num_heads": heads,
        "kv_heads": kv_heads,
        "branches": branches,
        "attention_type": kind,
        "dropout": 0,
    }
    if transformer:
        params.update(ff_dim=16, norm_first=int(norm_first))
    nodes = [
        {"id": "query", "op": "Input", "params": {"shape": [1, query, 8]}},
        {"id": "attn", "op": op, "params": params},
        {"id": "flatten", "op": "Flatten", "params": {}},
        {"id": "logits", "op": "Linear", "params": {"out_features": 2}},
        {"id": "output", "op": "Output", "params": {}},
    ]
    edges = [
        {"source": "query", "target": "attn"},
        {"source": "attn", "target": "flatten"},
        {"source": "flatten", "target": "logits"},
        {"source": "logits", "target": "output"},
    ]
    if kind == "cross":
        nodes.append({"id": "context", "op": "Input", "params": {"shape": [1, key, 8]}})
        edges[0]["targetPort"] = "query"
        edges.insert(0, {"source": "context", "target": "attn", "targetPort": "context"})
    return {"version": 1, "nodes": nodes, "edges": edges}


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AttentionInspectorRuntimeTests(unittest.TestCase):
    def request(self, graph, **kwargs):
        return TensorInferenceRequest(graph=graph, nodeIds=["attn"], attention={"attn": kwargs.pop("selection", {})}, **kwargs)

    def test_self_mqa_gqa_and_multibranch_return_real_matrices(self):
        for kind, kv_heads, branches in (("self", 4, 1), ("multi_query", 1, 1), ("grouped_query", 2, 2)):
            with self.subTest(kind=kind):
                graph = attention_graph(kind=kind, kv_heads=kv_heads, branches=branches)
                result = run_inference(self.request(graph, selection={"branch": 0, "head": 3}))
                item = result["attentions"][0]
                self.assertEqual(item["branches"], branches)
                self.assertEqual(item["kvHeads"], kv_heads)
                self.assertEqual(item["kvHead"], 3 // (4 // kv_heads))
                self.assertEqual(set(item["tensors"]), {"q", "k", "v", "scores", "probabilities", "headOutput", "branchOutput", "mergedOutput"})
                self.assertEqual(item["tensors"]["scores"]["shape"], [5, 5])

    def test_cross_attention_uses_context_length_and_projection_override(self):
        graph = attention_graph(kind="cross", kv_heads=1, query=4, key=7)
        # The graph has a K override whose source is a shorter sequence.  The
        # port-shape validator must use this actual override length.
        graph["nodes"].append({"id": "key_override", "op": "Input", "params": {"shape": [1, 3, 8]}})
        graph["edges"].extend({"source": "key_override", "target": "attn", "targetPort": f"b0:{role}0"} for role in ("k", "v"))
        result = run_inference(self.request(graph, selection={"branch": 0, "head": 0, "keyStart": 2}))
        item = result["attentions"][0]
        self.assertEqual(item["queryLength"], 4)
        self.assertEqual(item["keyLength"], 3)
        self.assertEqual(item["keyStart"], 2)

    def test_pre_and_post_norm_transformers_preserve_output_without_observer(self):
        for norm_first in (True, False):
            graph = attention_graph(transformer=True, norm_first=norm_first)
            model, _ = build_model(graph)
            x = torch.randn(1, 5, 8)
            expected = model(x)
            observed = model(x, attention_observer=lambda *_: None)
            torch.testing.assert_close(expected, observed)

    def test_scores_softmax_and_explicit_v_match_sdpa(self):
        graph = attention_graph(kind="grouped_query", kv_heads=2, heads=4)
        result = run_inference(self.request(graph, selection={"head": 2}))
        item = result["attentions"][0]
        probabilities = torch.tensor(item["tensors"]["probabilities"]["slice"]["values"])
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(5), atol=1e-5))
        scores = torch.tensor(item["tensors"]["scores"]["slice"]["values"])
        values = torch.tensor(item["tensors"]["v"]["slice"]["values"])
        expected = scores.softmax(-1) @ values
        actual = torch.tensor(item["tensors"]["headOutput"]["slice"]["values"])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_observer_does_not_change_output_or_global_rng(self):
        graph = attention_graph()
        model, _ = build_model(graph)
        x = torch.randn(1, 5, 8)
        expected = model(x)
        observed = model(x, attention_observer=lambda *_: None)
        torch.testing.assert_close(expected, observed)
        torch.manual_seed(77)
        expected_random = torch.rand(8)
        torch.manual_seed(77)
        run_inference(self.request(graph))
        torch.testing.assert_close(expected_random, torch.rand(8))

    def test_strict_selection_bounds_and_offsets_reject_before_build(self):
        graph = attention_graph(query=4, key=7)
        cases = (
            {"branch": 1},
            {"head": 4},
            {"queryStart": 4},
            {"keyStart": 7},
            {"unexpected": 1},
        )
        for selection in cases:
            with self.subTest(selection=selection):
                if "unexpected" in selection:
                    with self.assertRaises(ValueError):
                        TensorInferenceRequest(graph=graph, nodeIds=["attn"], attention={"attn": selection})
                else:
                    request = self.request(graph, selection=selection)
                    with patch("backend.training.build_model", side_effect=AssertionError("allocated too early")):
                        with self.assertRaises(ValueError):
                            run_inference(request)

    def test_extra_attention_scratch_budget_is_summed_before_model_build(self):
        from .graph import analyze_graph
        graph = attention_graph(query=256, heads=1)
        graph["nodes"].append({"id": "second", "op": "MultiHeadAttention", "params": dict(graph["nodes"][1]["params"])})
        graph["edges"][1] = {"source": "second", "target": "flatten"}
        graph["edges"].append({"source": "attn", "target": "second"})
        info = analyze_graph(graph)
        one_scratch = 256 * 256 * 64 + (256 * 8 * 4 + 256 * 8 * 2) * 32
        limit = info["activationBytesPerSample"] + one_scratch + one_scratch // 2
        request = TensorInferenceRequest(graph=graph, nodeIds=["attn", "second"], attention={"attn": {}, "second": {}})
        with patch("backend.inference.MAX_CAPTURE_BYTES", limit), patch("backend.training.build_model", side_effect=AssertionError("allocated too early")):
            with self.assertRaisesRegex(ValueError, "scratch"):
                run_inference(request)

    def test_full_native_kv_stacks_are_included_before_allocation(self):
        graph = attention_graph(kind="cross", heads=16, kv_heads=16, query=1, key=62500)
        graph["nodes"][0]["params"]["shape"][-1] = 64
        graph["nodes"][1]["params"]["embed_dim"] = 64
        graph["nodes"][-1]["params"]["shape"][-1] = 64
        with patch("backend.training.build_model", side_effect=AssertionError("allocated too early")):
            with self.assertRaisesRegex(ValueError, "scratch"):
                run_inference(self.request(graph))

    def test_full_selected_matrix_cap_is_not_evaded_by_a_small_window(self):
        graph = attention_graph(query=513, heads=1)
        with patch("backend.training.build_model", side_effect=AssertionError("allocated too early")):
            with self.assertRaisesRegex(ValueError, "262144"):
                run_inference(self.request(graph, selection={"queryStart": 512, "keyStart": 512}))

    def test_observed_report_preserves_normal_seeded_outputs(self):
        graph = attention_graph(kind="grouped_query", kv_heads=2, branches=2)
        plain = run_inference(TensorInferenceRequest(graph=graph, nodeIds=["attn", "output"], seed=19))
        observed = run_inference(TensorInferenceRequest(graph=graph, nodeIds=["attn", "output"], seed=19, attention={"attn": {"branch": 1, "head": 3}}))
        self.assertEqual(plain["tensors"], observed["tensors"])
        self.assertEqual(plain["attentions"], [])

    def test_actual_branch_outputs_merge_and_transformer_norm_inputs(self):
        for norm_first in (True, False):
            graph = attention_graph(kind="cross", kv_heads=2, branches=2, transformer=True, norm_first=norm_first)
            graph["nodes"].append({"id": "override", "op": "Input", "params": {"shape": [1, 5, 8]}})
            graph["edges"].append({"source": "override", "target": "attn", "targetPort": "b1:q3"})
            model, _ = build_model(graph); model.eval()
            inputs = {"query": torch.randn(1, 5, 8), "context": torch.randn(1, 7, 8), "override": torch.randn(1, 5, 8)}
            seen = {}
            def observe(node, stage, branch, tensors):
                seen[(stage, branch)] = tensors
            with torch.no_grad():
                plain = model(inputs)
                observed = model(inputs, attention_observer=observe)
            torch.testing.assert_close(plain, observed)
            layer = model.layers["attn"]; branch = layer.attention.branches[1]
            query = layer.norm1(inputs["override"]) if norm_first else inputs["override"]
            expected_q = torch.nn.functional.linear(query, branch.q_proj.weight[6:8], branch.q_proj.bias[6:8])
            torch.testing.assert_close(seen[("q", 1)]["q"][:, 3], expected_q)
            expected_k = branch.k_proj(inputs["context"]).reshape(1, 7, 2, 2).transpose(1, 2)
            torch.testing.assert_close(seen[("k", 1)]["k"], expected_k)
            merged = (seen[("output", 0)]["output"] + seen[("output", 1)]["output"]) / 2
            torch.testing.assert_close(seen[("merged", -1)]["output"], merged)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA not available")
    def test_trained_attention_state_and_cpu_cuda_parity(self):
        import threading
        from .training import train
        from .trained_models import TrainedModelStore
        graph = attention_graph(kind="grouped_query", kv_heads=2)
        store = TrainedModelStore(); messages = []
        config = {"epochs": 2, "learningRate": 0.01, "batchSize": 16, "samples": 64, "device": "cpu", "dataset": "synthetic", "validationFraction": 0.2, "earlyStopping": False, "patience": 3}
        train(graph, config, messages.append, threading.Event(), retain_model=store.save)
        model_id = messages[-1]["model"]["modelId"]
        with patch("backend.trained_models.trained_models", store):
            cpu = run_inference(self.request(graph, modelId=model_id))
            gpu = run_inference(self.request(graph, modelId=model_id, device="cuda"))
        self.assertEqual(cpu["weights"], "trained")
        self.assertEqual(cpu["model"], gpu["model"])
        for name in cpu["attentions"][0]["tensors"]:
            torch.testing.assert_close(torch.tensor(cpu["attentions"][0]["tensors"][name]["slice"]["values"]), torch.tensor(gpu["attentions"][0]["tensors"][name]["slice"]["values"]), atol=1e-5, rtol=1e-4)

    def test_window_offsets_and_nonfinite_json_contract(self):
        graph = attention_graph(query=20)
        result = run_inference(self.request(graph, selection={"queryStart": 3, "keyStart": 4}))
        scores = result["attentions"][0]["tensors"]["scores"]
        self.assertEqual((scores["rowStart"], scores["columnStart"]), (3, 4))
        self.assertEqual(scores["slice"]["shape"], [20, 20])
        self.assertTrue(scores["slice"]["truncated"])
        json.dumps(result, allow_nan=False)
        from .inference import _attention_matrix
        nonfinite = _attention_matrix("scores", torch.tensor([[float("nan"), float("inf"), -float("inf")]]), 0, 1, torch)
        self.assertEqual(nonfinite["nonFiniteCount"], 3)
        self.assertEqual(nonfinite["slice"]["values"], [[None, None]])
        json.dumps(nonfinite, allow_nan=False)


class AttentionInspectorRequestTests(unittest.TestCase):
    def test_attention_is_optional_and_extra_fields_are_forbidden(self):
        graph = attention_graph()
        request = TensorInferenceRequest(graph=graph, nodeIds=["attn"])
        self.assertEqual(request.attention, {})
        with self.assertRaises(ValueError):
            TensorInferenceRequest(graph=graph, nodeIds=["attn"], attention={"attn": {"head": 0, "extra": 1}})


if __name__ == "__main__":
    unittest.main()
