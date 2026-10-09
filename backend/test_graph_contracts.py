import json
import math
import unittest
from pathlib import Path
from .graph import MAX_REPEAT, analyze_graph


FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "analysis-contracts.json"


def folded_linear_graph(repeat=None):
    nodes = [
        {"id": "n0", "op": "Input", "params": {"shape": [2, 4]}},
        {"id": "n1", "op": "Linear", "params": {"out_features": 4}},
        {"id": "n2", "op": "ReLU", "params": {}},
        {"id": "n3", "op": "Linear", "params": {"out_features": 2}},
        {"id": "n4", "op": "Output", "params": {}},
    ]
    if repeat is not None: nodes[1]["repeat"] = repeat
    return {"version": 1, "nodes": nodes, "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(4)]}


class GraphContractTests(unittest.TestCase):
    def test_shared_analysis_contracts(self):
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
        for case in cases:
            with self.subTest(case=case["id"]):
                expected = case["expected"]
                if expected["valid"]:
                    info = analyze_graph(case["graph"])
                    self.assertEqual(info["shapes"][info["output"]], expected["outputShape"])
                    self.assertEqual(info["totalParameters"], expected["parameters"])
                else:
                    with self.assertRaises(ValueError): analyze_graph(case["graph"])


class RepeatFoldTests(unittest.TestCase):
    def test_absent_repeat_matches_the_single_instance_baseline(self):
        baseline = analyze_graph(folded_linear_graph())
        self.assertEqual(baseline["totalParameters"], 30)
        self.assertEqual(baseline["parameters"], {"n0": 0, "n1": 20, "n2": 0, "n3": 10, "n4": 0})
        # An explicit repeat of 1 must be indistinguishable from leaving the field
        # off in every computed output (the analyzer echoes the input nodes back,
        # which is the only place the two graphs differ).
        computed = lambda info: {key: info[key] for key in ("order", "shapes", "portShapes", "parameters", "totalParameters", "activationBytesPerSample")}
        self.assertEqual(computed(analyze_graph(folded_linear_graph(1))), computed(baseline))

    def test_repeat_scales_parameters_and_activation_but_not_shapes(self):
        baseline = analyze_graph(folded_linear_graph())
        for repeat in (2, 3, 17, MAX_REPEAT):
            with self.subTest(repeat=repeat):
                info = analyze_graph(folded_linear_graph(repeat))
                # The folded node reports exactly the single-instance count times N.
                self.assertEqual(info["parameters"]["n1"], 20 * repeat)
                self.assertEqual(info["parameters"]["n3"], 10)
                self.assertEqual(info["totalParameters"], 20 * repeat + 10)
                # Instances are chained, so every shape is unchanged.
                self.assertEqual(info["shapes"]["n1"], baseline["shapes"]["n1"])
                self.assertEqual(info["shapes"]["n4"], baseline["shapes"]["n4"])
                # Activation memory covers N copies of the folded node.
                self.assertEqual(
                    info["activationBytesPerSample"],
                    baseline["activationBytesPerSample"] + math.prod(baseline["shapes"]["n1"][1:]) * 4 * (repeat - 1),
                )

    def test_repeat_must_be_an_integer_in_range(self):
        # 1.0 is deliberately absent: the wire format is produced by JavaScript, where
        # JSON.parse("1.0") yields the integer 1, so the two sides cannot disagree on it.
        for repeat in (0, -1, 1025, 2.5, True, "2"):
            with self.subTest(repeat=repeat):
                with self.assertRaisesRegex(ValueError, "repeat"):
                    analyze_graph(folded_linear_graph(repeat))

    def test_integral_floats_are_folded_like_integers(self):
        # Python's json.loads turns 1.0 into a float; JavaScript's JSON.parse turns it into 1.
        # Accepting integral floats keeps the same document valid on both sides.
        for repeat in (1.0, 2.0, 3.0):
            with self.subTest(repeat=repeat):
                info = analyze_graph(folded_linear_graph(repeat))
                self.assertEqual(info["totalParameters"], 20 * int(repeat) + 10)


if __name__ == "__main__": unittest.main()
