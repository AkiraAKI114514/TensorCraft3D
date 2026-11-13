import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze } from './analysis';
import { generatePython } from './export';
import type { Graph, Layer } from './types';

// The exported module and the re-imported graph run on the same weights (and the same gate bias
// buffer) and input, and must agree exactly, so routing, shared experts and every parameter have to
// survive both directions.
const SCRIPT = `import json, sys, torch
from backend.pytorch_import import import_pytorch
from backend.training import build_model
p = json.load(sys.stdin)
r = import_pytorch(p["code"])
assert r["graph"] is not None, r["diagnostics"]
nodes = r["graph"]["nodes"] + [n for s in (r["graph"].get("subgraphs") or {}).values() for n in s["nodes"]]
moe = [n for n in nodes if n["op"] == "MoE"]
assert len(moe) == 1, [n["op"] for n in nodes]
assert moe[0]["params"] == p["params"], (moe[0]["params"], p["params"])
groups = [n for n in r["graph"]["nodes"] if n["op"] == "Group"]
assert [g.get("repeat") for g in groups] == p["groups"], [g.get("repeat") for g in groups]
imported, info = build_model(r["graph"])
assert info["totalParameters"] == p["parameters"], (info["totalParameters"], p["parameters"])
namespace = {"__name__": "moe_export"}
exec(p["code"], namespace)
exported = namespace["VisualModel"]().eval()
imported = imported.eval()
# Names differ (node ids, folded instances), so the tensors pair up by position; the gate bias is a
# buffer, so it is in the state dict and gets a random value to make it count.
left, right = list(exported.state_dict().items()), list(imported.state_dict().values())
assert len(left) == len(right), (len(left), len(right))
assert all(a.shape == b.shape for (_, a), b in zip(left, right))
with torch.no_grad():
    for (key, a), b in zip(left, right):
        if key.endswith("gate.bias"): a.copy_(torch.randn_like(a) * 0.2)
        b.copy_(a)
    x = torch.randn(2, 4, 16)
    assert torch.equal(exported(x), imported(x)), "the export changed the computation"
print("matched")
`;

const PARAMS = { experts: 4, top_k: 2, shared_experts: 1, expert_dim: 8, score_func: 'sigmoid', route_scale: 2.5, groups: 2, topk_groups: 1, gate_bias: 1 };
const node = (id: string, op: string, params: Layer['params'] = {}, repeat?: number): Layer =>
  ({ id, name: id, op: op as Layer['op'], params, position: { x: 0, y: 0 }, ...(repeat === undefined ? {} : { repeat }) });
const moeGraph = (params: Layer['params'], repeat?: number): Graph => ({
  version: 1, name: 'Mixture of experts',
  nodes: [node('n0', 'Input', { shape: [2, 4, 16] }), node('n1', 'MoE', params, repeat), node('n2', 'Output')],
  edges: [{ id: 'e0', source: 'n0', target: 'n1' }, { id: 'e1', source: 'n1', target: 'n2' }]
});

function roundtrip(graph: Graph, groups: number[]) {
  const analysis = analyze(graph), code = generatePython(graph);
  expect(analysis.valid, JSON.stringify(analysis.diagnostics)).toBe(true);
  return { code, result: execFileSync('.venv/Scripts/python.exe', ['-c', SCRIPT], { input: JSON.stringify({ code, params: graph.nodes[1].params, parameters: analysis.parameters, groups }), encoding: 'utf8', timeout: 60000 }) };
}

describe('mixture-of-experts export roundtrip', () => {
  it('writes every routing parameter and reads it back as the same computation', () => {
    const { code, result } = roundtrip(moeGraph(PARAMS), []);
    expect(code).toContain('TensorLabMoE(dim=16, experts=4, top_k=2, shared_experts=1, expert_dim=8, score_func="sigmoid", route_scale=2.5, groups=2, topk_groups=1, gate_bias=True)');
    expect(code).toContain('class TensorLabMoE(nn.Module)');
    expect(result).toContain('matched');
  }, 60000);

  it('keeps a softmax gate without bias or shared experts', () => {
    const { code, result } = roundtrip(moeGraph({ ...PARAMS, score_func: 'softmax', shared_experts: 0, groups: 1, gate_bias: 0 }), []);
    expect(code).toContain('gate_bias=False');
    expect(result).toContain('matched');
  }, 60000);

  it('folds a repeated layer back into one block of two instances', () => {
    expect(roundtrip(moeGraph(PARAMS, 2), [2]).result).toContain('matched');
  }, 60000);
});
