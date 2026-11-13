import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze } from './analysis';
import { generatePython } from './export';
import type { Graph, Layer } from './types';

// The exported module and the re-imported graph run on the same weights and input and must agree
// exactly, so every MLA parameter, the mask flag included, has to survive both directions.
const SCRIPT = `import json, sys, torch
from backend.pytorch_import import import_pytorch
from backend.training import build_model
p = json.load(sys.stdin)
r = import_pytorch(p["code"])
assert r["graph"] is not None, r["diagnostics"]
nodes = r["graph"]["nodes"] + [n for s in (r["graph"].get("subgraphs") or {}).values() for n in s["nodes"]]
latent = [n for n in nodes if n["op"] == "LatentAttention"]
assert len(latent) == 1, [n["op"] for n in nodes]
assert latent[0]["params"] == p["params"], (latent[0]["params"], p["params"])
groups = [n for n in r["graph"]["nodes"] if n["op"] == "Group"]
assert [g.get("repeat") for g in groups] == p["groups"], [g.get("repeat") for g in groups]
imported, info = build_model(r["graph"])
assert info["totalParameters"] == p["parameters"], (info["totalParameters"], p["parameters"])
namespace = {"__name__": "latent_export"}
exec(p["code"], namespace)
exported = namespace["VisualModel"]().eval()
imported = imported.eval()
left, right = list(exported.parameters()), list(imported.parameters())
assert len(left) == len(right), (len(left), len(right))
with torch.no_grad():
    for a, b in zip(left, right): b.copy_(a)
    x = torch.randn(2, 6, 32)
    assert torch.equal(exported(x), imported(x)), "the export changed the computation"
print("matched")
`;

const PARAMS = { embed_dim: 32, num_heads: 4, q_lora_rank: 16, kv_lora_rank: 8, qk_nope_head_dim: 8, qk_rope_head_dim: 4, v_head_dim: 8, rope_theta: 10000, causal: 1, eps: 1e-6 };
const node = (id: string, op: string, params: Layer['params'] = {}, repeat?: number): Layer =>
  ({ id, name: id, op: op as Layer['op'], params, position: { x: 0, y: 0 }, ...(repeat === undefined ? {} : { repeat }) });
const latentGraph = (params: Layer['params'], repeat?: number): Graph => ({
  version: 1, name: 'Latent attention',
  nodes: [node('n0', 'Input', { shape: [2, 6, 32] }), node('n1', 'LatentAttention', params, repeat), node('n2', 'Output')],
  edges: [{ id: 'e0', source: 'n0', target: 'n1' }, { id: 'e1', source: 'n1', target: 'n2' }]
});

function roundtrip(graph: Graph, groups: number[]) {
  const analysis = analyze(graph), code = generatePython(graph);
  expect(analysis.valid, JSON.stringify(analysis.diagnostics)).toBe(true);
  const params = graph.nodes[1].params;
  return { code, result: execFileSync('.venv/Scripts/python.exe', ['-c', SCRIPT], { input: JSON.stringify({ code, params, parameters: analysis.parameters, groups }), encoding: 'utf8', timeout: 60000 }) };
}

describe('latent attention export roundtrip', () => {
  it('writes every parameter and reads it back as the same computation', () => {
    const { code, result } = roundtrip(latentGraph(PARAMS), []);
    expect(code).toContain('TensorLabLatentAttention(embed_dim=32, num_heads=4, q_lora_rank=16, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8, rope_theta=10000, causal=True, eps=0.000001)');
    expect(code).toContain('class TensorLabLatentAttention(nn.Module)');
    expect(result).toContain('matched');
  }, 60000);

  it('keeps an unmasked layer without a query latent unmasked', () => {
    const { code, result } = roundtrip(latentGraph({ ...PARAMS, q_lora_rank: 0, causal: 0 }), []);
    expect(code).toContain('q_lora_rank=0');
    expect(code).toContain('causal=False');
    expect(result).toContain('matched');
  }, 60000);

  it('folds a repeated layer back into one block of two instances', () => {
    expect(roundtrip(latentGraph(PARAMS, 2), [2]).result).toContain('matched');
  }, 60000);
});
