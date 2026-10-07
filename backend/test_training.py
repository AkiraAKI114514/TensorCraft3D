import unittest
import threading
from .graph import analyze_graph
from .training import build_model, load_data, train


def mlp_graph():
    specs = [("Input", {"shape": [1, 4]}), ("Linear", {"out_features": 8}), ("ReLU", {}), ("Linear", {"out_features": 2}), ("Output", {})]
    nodes = [{"id": f"layer_{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
    return {"version": 1, "nodes": nodes, "edges": [{"source": f"layer_{i}", "target": f"layer_{i+1}"} for i in range(4)]}


def transformer_graph(op="Transformer"):
    specs = [("Input", {"shape": [1, 4, 8]}), (op, {"embed_dim": 8, "num_heads": 2, "ff_dim": 16, "dropout": 0}), ("Flatten", {}), ("Linear", {"out_features": 2}), ("Output", {})]
    nodes = [{"id": f"layer_{i}", "op": kind, "params": params} for i, (kind, params) in enumerate(specs)]
    return {"version": 1, "nodes": nodes, "edges": [{"source": f"layer_{i}", "target": f"layer_{i+1}"} for i in range(4)]}


def cross_graph(op="MultiHeadAttention"):
    graph = transformer_graph(op)
    graph["nodes"][1]["params"].update(attention_type="cross", kv_heads=1, branches=2)
    graph["nodes"].append({"id": "context", "op": "Input", "params": {"shape": [1, 7, 8]}})
    graph["edges"][0]["targetPort"] = "query"
    graph["edges"].insert(0, {"source": "context", "target": "layer_1", "targetPort": "context"})
    return graph


def bilinear_graph(in2_features=16, second_shape=(4, 16)):
    params = {"out_features": 10}
    if in2_features is not None: params["in2_features"] = in2_features
    specs = [("Input", {"shape": [4, 8]}), ("Bilinear", params), ("Output", {})]
    nodes = [{"id": f"layer_{i}", "op": op, "params": params} for i, (op, params) in enumerate(specs)]
    nodes.append({"id": "second", "op": "Input", "params": {"shape": list(second_shape)}})
    return {"version": 1, "nodes": nodes, "edges": [{"source": "layer_0", "target": "layer_1"}, {"source": "second", "target": "layer_1"}, {"source": "layer_1", "target": "layer_2"}]}


class TrainingTests(unittest.TestCase):
    def test_bilinear_uses_in2_features_for_shape_and_the_real_module(self):
        import torch
        for in2_features, width in ((16, 16), (None, 16), (7, 7)):
            with self.subTest(in2_features=in2_features, width=width):
                graph = bilinear_graph(in2_features, second_shape=(4, width))
                info = analyze_graph(graph)
                model, _ = build_model(graph)
                self.assertEqual(info["shapes"]["layer_1"], [4, 10])
                self.assertEqual(model.layers["layer_1"].in2_features, width)
                self.assertEqual(info["totalParameters"], sum(p.numel() for p in model.parameters()))
                self.assertEqual(info["totalParameters"], 8 * width * 10 + 10)
                output = model({"layer_0": torch.randn(4, 8), "second": torch.randn(4, width)})
                self.assertEqual(tuple(output.shape), (4, 10))

    def test_bilinear_rejects_in2_features_mismatch(self):
        for graph in (bilinear_graph(in2_features=8), bilinear_graph(in2_features=None, second_shape=(4, 8))):
            with self.subTest(params=graph["nodes"][1]["params"]):
                with self.assertRaisesRegex(ValueError, "in2_features"): analyze_graph(graph)
                with self.assertRaisesRegex(ValueError, "in2_features"): build_model(graph)

    def test_bilinear_rejects_batch_and_rank_mismatch(self):
        for shape, message in (((5, 16), "batch"), ((4, 2, 16), "2D")):
            with self.subTest(shape=shape):
                graph = bilinear_graph(second_shape=shape)
                with self.assertRaisesRegex(ValueError, message): analyze_graph(graph)
                with self.assertRaisesRegex(ValueError, message): build_model(graph)

    def test_shape_and_parameter_contract(self):
        import torch
        model, info = build_model(mlp_graph())
        self.assertEqual(info["totalParameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(tuple(model(torch.randn(5, 4)).shape), (5, 2))

    def test_linear_preserves_leading_dimensions_and_matches_pytorch(self):
        import torch
        from torch import nn
        for shape in ([2, 101], [64, 6, 101], [64, 6, 69], [2, 3, 4, 5], [2, 3, 4, 5, 6]):
            with self.subTest(shape=shape):
                graph = mlp_graph()
                graph["nodes"][0]["params"]["shape"] = list(shape)
                model, info = build_model(graph)
                reference = nn.Sequential(nn.Linear(shape[-1], 8), nn.ReLU(), nn.Linear(8, 2))
                reference[0].load_state_dict(model.layers["layer_1"].state_dict())
                reference[2].load_state_dict(model.layers["layer_3"].state_dict())
                x = torch.randn(*shape, requires_grad=True)
                output = model(x)
                self.assertEqual(list(output.shape), [*shape[:-1], 2])
                self.assertEqual(info["shapes"]["layer_4"], list(output.shape))
                self.assertEqual(info["totalParameters"], sum(p.numel() for p in model.parameters()))
                torch.testing.assert_close(output, reference(x))
                output.square().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_rejects_huge_model_before_allocation(self):
        graph = mlp_graph(); graph["nodes"][0]["params"]["shape"] = [1, 65536]; graph["nodes"][1]["params"]["out_features"] = 65536
        with self.assertRaises(ValueError): analyze_graph(graph)

    def test_training_emits_measured_metrics(self):
        messages = []
        config = {"epochs": 3, "learningRate": 0.01, "batchSize": 16, "samples": 64, "device": "cpu", "dataset": "synthetic", "validationFraction": 0.2, "earlyStopping": False, "patience": 5}
        train(mlp_graph(), config, messages.append, threading.Event())
        metrics = [m["metric"] for m in messages if m["type"] == "metric"]
        self.assertEqual(len(metrics), 3)
        self.assertLess(metrics[-1]["trainLoss"], metrics[0]["trainLoss"])
        self.assertTrue(metrics[0]["layerGradients"])
        self.assertEqual(metrics[0]["source"], "training")
        self.assertEqual(messages[-1], {"type": "done", "reason": "completed"})

    def test_stop_is_honored(self):
        stop = threading.Event(); stop.set(); messages = []
        config = {"epochs": 2, "learningRate": 0.01, "batchSize": 16, "samples": 64, "device": "cpu", "dataset": "synthetic", "validationFraction": 0.2, "earlyStopping": False, "patience": 5}
        train(mlp_graph(), config, messages.append, stop)
        self.assertEqual(messages, [{"type": "done", "reason": "stopped"}])

    def test_transformer_and_attention_parameter_contracts(self):
        import torch
        for op in ("Transformer", "MultiHeadAttention"):
            model, info = build_model(transformer_graph(op))
            self.assertEqual(info["totalParameters"], sum(p.numel() for p in model.parameters()))
            value = model(torch.randn(3, 4, 8))
            self.assertEqual(tuple(value.shape), (3, 2))
            value.sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layers["layer_1"].parameters()))

    def test_attention_validation_before_allocation(self):
        graph = transformer_graph(); graph["nodes"][1]["params"]["num_heads"] = 3
        with self.assertRaises(ValueError): analyze_graph(graph)
        graph["nodes"][1]["params"]["num_heads"] = 2; graph["nodes"][0]["params"]["shape"] = [1, 4096, 8]
        with self.assertRaises(ValueError): analyze_graph(graph)

    def test_attention_defaults_and_visualization_parameter(self):
        for op in ("Transformer", "MultiHeadAttention"):
            graph = transformer_graph(op)
            del graph["nodes"][1]["params"]["num_heads"]
            model, info = build_model(graph)
            layer = model.layers["layer_1"]
            attention = layer.attention if op == "Transformer" else layer
            self.assertEqual(attention.num_heads, 1)
            graph["nodes"][1]["params"]["qkv_count"] = 6
            self.assertEqual(analyze_graph(graph)["totalParameters"], info["totalParameters"])
            for count in (2, 65, 4.5):
                graph["nodes"][1]["params"]["qkv_count"] = count
                self.assertEqual(analyze_graph(graph)["totalParameters"], info["totalParameters"])

    def test_transformer_real_sequence_training(self):
        messages = []
        config = {"epochs": 2, "learningRate": 0.01, "batchSize": 16, "samples": 64, "device": "cpu", "dataset": "synthetic", "validationFraction": 0.2, "earlyStopping": False, "patience": 5}
        train(transformer_graph(), config, messages.append, threading.Event())
        metrics = [m["metric"] for m in messages if m["type"] == "metric"]
        self.assertEqual(len(metrics), 2)
        self.assertGreater(metrics[0]["layerGradients"]["layer_1"], 0)
        self.assertEqual(messages[-1]["reason"], "completed")

    def test_architectures_forward_backward_and_exact_parameters(self):
        import torch
        for op in ("Transformer", "MultiHeadAttention"):
            for kind, heads, kv_heads, branches in (("self", 4, 4, 1), ("multi_query", 4, 1, 1), ("grouped_query", 4, 2, 3), ("multi_branch", 4, 4, 2)):
                with self.subTest(op=op, kind=kind):
                    graph = transformer_graph(op)
                    graph["nodes"][1]["params"].update(attention_type=kind, num_heads=heads, kv_heads=kv_heads, branches=branches)
                    model, info = build_model(graph)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), info["totalParameters"])
                    model(torch.randn(3, 4, 8)).square().sum().backward()
                    attention = model.layers["layer_1"].attention if op == "Transformer" else model.layers["layer_1"]
                    self.assertEqual(attention.branches[0].k_proj.out_features, 8 // heads * kv_heads)
                    for p in attention.parameters():
                        self.assertIsNotNone(p.grad)
                        self.assertTrue(torch.isfinite(p.grad).all())
                        self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_cross_roles_lengths_and_gradients(self):
        import torch
        for op in ("Transformer", "MultiHeadAttention"):
            graph = cross_graph(op)
            model, info = build_model(graph)
            self.assertEqual(info["incoming"]["layer_1"], ["layer_0", "context"])
            self.assertEqual(info["shapes"]["layer_1"], [1, 4, 8])
            self.assertEqual(info["totalParameters"], sum(p.numel() for p in model.parameters()))
            inputs = {"layer_0": torch.randn(3, 4, 8, requires_grad=True), "context": torch.randn(3, 7, 8, requires_grad=True)}
            model(inputs).square().sum().backward()
            for value in inputs.values(): self.assertGreater(value.grad.abs().sum().item(), 0)
            with self.assertRaises(ValueError): model(inputs["layer_0"])
            with self.assertRaises(ValueError): model({"layer_0": inputs["layer_0"]})

    def test_shared_kv_attention_matches_explicit_softmax(self):
        import torch
        from .attention import TensorLabAttention
        torch.manual_seed(11)
        for heads, kv_heads, kind in ((4, 1, "multi_query"), (4, 2, "grouped_query"), (4, 4, "self")):
            layer = TensorLabAttention(8, heads, kv_heads, dropout=0, attention_type=kind, branches=2).eval()
            x = torch.randn(2, 5, 8)
            results = []
            for branch in layer.branches:
                q = branch.q_proj(x).view(2, 5, heads, 2).transpose(1, 2)
                k = branch.k_proj(x).view(2, 5, kv_heads, 2).transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
                v = branch.v_proj(x).view(2, 5, kv_heads, 2).transpose(1, 2).repeat_interleave(heads // kv_heads, 1)
                weights = (q @ k.transpose(-2, -1) / 2 ** 0.5).softmax(-1)
                result = (weights @ v).transpose(1, 2).reshape(2, 5, 8)
                results.append(branch.out_proj(result))
            torch.testing.assert_close(layer(x), sum(results) / 2)

    def test_custom_attention_matches_standard_pytorch_modules(self):
        import torch
        from torch import nn
        from .attention import TensorLabAttention, TensorLabTransformer
        x = torch.randn(2, 5, 8)
        for transformer in (False, True):
            custom = TensorLabTransformer(8, 2, 2, 16, dropout=0) if transformer else TensorLabAttention(8, 2, 2, dropout=0)
            reference = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, activation="gelu", batch_first=True, norm_first=True) if transformer else nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)
            attention = custom.attention if transformer else custom
            branch = attention.branches[0]
            ref_attn = reference.self_attn if transformer else reference
            with torch.no_grad():
                ref_attn.in_proj_weight.copy_(torch.cat([branch.q_proj.weight, branch.k_proj.weight, branch.v_proj.weight]))
                ref_attn.in_proj_bias.copy_(torch.cat([branch.q_proj.bias, branch.k_proj.bias, branch.v_proj.bias]))
            ref_attn.out_proj.load_state_dict(branch.out_proj.state_dict())
            if transformer:
                reference.norm1.load_state_dict(custom.norm1.state_dict()); reference.norm2.load_state_dict(custom.norm2.state_dict())
                reference.linear1.load_state_dict(custom.ffn[0].state_dict()); reference.linear2.load_state_dict(custom.ffn[3].state_dict())
            custom.eval(); reference.eval()
            expected = reference(x) if transformer else reference(x, x, x, need_weights=False)[0]
            torch.testing.assert_close(custom(x), expected)

    def test_architecture_errors_rejected_before_allocation(self):
        invalid = [dict(attention_type="bad"), dict(attention_type="multi_query", kv_heads=2), dict(attention_type="grouped_query", num_heads=4, kv_heads=3), dict(attention_type="self", kv_heads=1), dict(branches=9), dict(branches=1.5), dict(kv_heads=True), dict(attention_type="multi_branch", branches=1)]
        for params in invalid:
            graph = transformer_graph(); graph["nodes"][1]["params"].update(params)
            with self.subTest(params=params), self.assertRaises(ValueError): analyze_graph(graph)
        for variant in ("missing", "duplicate", "batch", "embed", "rank"):
            graph = cross_graph()
            if variant == "missing": graph["edges"].pop(0)
            elif variant == "duplicate": graph["edges"][0]["targetPort"] = "query"
            elif variant == "batch": graph["nodes"][-1]["params"]["shape"] = [2, 7, 8]
            elif variant == "embed": graph["nodes"][-1]["params"]["shape"] = [1, 7, 4]
            else: graph["nodes"][-1]["params"]["shape"] = [1, 8]
            with self.subTest(variant=variant), self.assertRaises(ValueError): analyze_graph(graph)

    def test_cross_real_training_and_csv_input_order(self):
        import torch
        graph = cross_graph("Transformer"); messages = []
        config = {"epochs": 2, "learningRate": 0.01, "batchSize": 16, "samples": 64, "device": "cpu", "dataset": "synthetic", "validationFraction": 0.2, "earlyStopping": False, "patience": 5}
        train(graph, config, messages.append, threading.Event())
        metrics = [m["metric"] for m in messages if m["type"] == "metric"]
        self.assertEqual(len(metrics), 2)
        self.assertGreater(metrics[0]["layerGradients"]["layer_1"], 0)
        info = analyze_graph(graph)
        csv = "\n".join(",".join(map(str, list(range(88)) + [i % 2])) for i in range(16))
        data, labels = load_data(info, {**config, "dataset": "csv", "csv": csv}, 2)
        self.assertEqual(tuple(data["layer_0"].shape), (16, 4, 8))
        self.assertEqual(tuple(data["context"].shape), (16, 7, 8))
        torch.testing.assert_close(data["layer_0"][0].flatten(), torch.arange(32).float())
        torch.testing.assert_close(data["context"][0].flatten(), torch.arange(32, 88).float())
        self.assertEqual(labels.tolist(), [i % 2 for i in range(16)])

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA not available in this PyTorch installation")
    def test_shared_cross_attention_cuda_backward(self):
        import torch
        model, _ = build_model(cross_graph("Transformer")); model.cuda()
        result = model({"layer_0": torch.randn(2, 4, 8, device="cuda"), "context": torch.randn(2, 7, 8, device="cuda")})
        result.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__": unittest.main()
