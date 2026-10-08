import unittest

from .graph import analyze_graph
from .training import build_model


def graph(op, shape, params=None):
    params = params or {}
    return {
        "version": 1,
        "name": f"{op} rank contract",
        "nodes": [
            {"id": "input", "name": "Input", "op": "Input", "params": {"shape": shape}},
            {"id": "layer", "name": op, "op": op, "params": params},
            {"id": "output", "name": "Output", "op": "Output", "params": {}},
        ],
        "edges": [
            {"id": "in", "source": "input", "target": "layer"},
            {"id": "out", "source": "layer", "target": "output"},
        ],
    }


class ChannelRankTests(unittest.TestCase):
    def test_batchnorm_rank_contracts_match_build_and_forward(self):
        import torch

        cases = [
            ("BatchNorm1d", [1, 4]),
            ("BatchNorm1d", [1, 4, 6]),
            ("BatchNorm2d", [1, 4, 3, 3]),
            ("BatchNorm3d", [1, 4, 2, 3, 3]),
        ]
        for op, shape in cases:
            with self.subTest(op=op, shape=shape):
                graph_value = graph(op, shape)
                analysis = analyze_graph(graph_value)
                model, info = build_model(graph_value)
                self.assertEqual(info["totalParameters"], sum(parameter.numel() for parameter in model.parameters()))
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(*shape))
                self.assertEqual(list(output.shape), shape)

    def test_instance_norm_1d_requires_sequence_axis(self):
        with self.assertRaises(ValueError):
            analyze_graph(graph("InstanceNorm1d", [2, 4]))

    def test_rank_one_channel_smoke_paths(self):
        import torch

        for op, params, expected in [
            ("ConvTranspose1d", {}, [2, 16, 16]),
            ("MaxPool1d", {}, [2, 4, 4]),
            ("AdaptiveAvgPool1d", {"output_size": 2}, [2, 4, 2]),
        ]:
            with self.subTest(op=op):
                graph_value = graph(op, [2, 4, 8], params)
                model, info = build_model(graph_value)
                self.assertEqual(info["shapes"]["output"], expected)
                model.eval()
                with torch.no_grad():
                    self.assertEqual(list(model(torch.randn(2, 4, 8)).shape), expected)


if __name__ == "__main__":
    unittest.main()
