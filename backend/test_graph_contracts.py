import json
import unittest
from pathlib import Path
from .graph import analyze_graph


FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "analysis-contracts.json"


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


if __name__ == "__main__": unittest.main()
