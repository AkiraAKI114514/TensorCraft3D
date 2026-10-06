import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze, validateGraph } from './analysis';
import { generatePython } from './export';
import { PRESETS } from './presets';
import type { Graph } from './types';

function roundtrip(graph: Graph) {
  const script = `import json,sys,ast,torch\nfrom backend.pytorch_import import import_pytorch\nfrom backend.training import build_model\np=json.load(sys.stdin)\nr=import_pytorch(p["code"])\nassert r["graph"] is not None,r["diagnostics"]\na,ia=build_model(p["graph"]);b,ib=build_model(r["graph"])\nfor n in r["graph"]["nodes"]:\n if n["id"] in b.layers:\n  original_id=ast.literal_eval(n["name"].split("self.layers",1)[1][1:-1])\n  b.layers[n["id"]].load_state_dict(a.layers[original_id].state_dict())\na.eval();b.eval()\nx={k:torch.randn(*ia["shapes"][k]) for k in ia["inputs"]}\ny={i["id"]:x[i["name"]] for i in r["inputs"]}\nassert ia["totalParameters"]==ib["totalParameters"]\nassert torch.equal(a(x),b(y)),"Roundtrip changed computation"\nprint(json.dumps(r["graph"]))\n`;
  return JSON.parse(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify({ graph, code: generatePython(graph) }), encoding: 'utf8', timeout: 30000 })) as Graph;
}

describe('PyTorch code imports', () => {
  it.each(['cnn', 'mlp', 'residual', 'transformer', 'mqa', 'gqa', 'cross_attention'])('reimports exported %s with identical computation', preset => {
    const original = PRESETS[preset](), imported = roundtrip(original);
    expect(validateGraph(imported)).toEqual(imported);
    expect(analyze(imported).valid).toBe(true);
    expect(analyze(imported).parameters).toBe(analyze(original).parameters);
  }, 30000);

  it('preserves Q/K/V overrides and projection outputs', () => {
    const graph = PRESETS.mqa();
    graph.nodes.push({ id: 'extra', name: 'Extra input', op: 'Input', params: { shape: [1, 16, 64] }, position: { x: 0, y: 300 } });
    for (const port of ['q0', 'k0', 'v0']) graph.edges.push({ id: `override_${port}`, source: 'extra', target: 'layer_1', targetPort: `b0:${port}` });
    graph.nodes.push({ id: 'projection', name: 'Q output', op: 'Flatten', params: {}, position: { x: 420, y: 300 } });
    graph.edges.push({ id: 'projected', source: 'layer_1', sourcePort: 'b0:q0', target: 'projection' });
    graph.nodes.push({ id: 'concat', name: 'Join', op: 'Concat', params: { dim: 1 }, position: { x: 500, y: 100 } });
    graph.edges = graph.edges.filter(e => e.target !== 'layer_3');
    graph.edges.push({ id: 'base', source: 'layer_2', target: 'concat' }, { id: 'port', source: 'projection', target: 'concat' }, { id: 'head', source: 'concat', target: 'layer_3' });
    const imported = roundtrip(graph);
    expect(imported.edges.filter(e => e.targetPort?.startsWith('b0:'))).toHaveLength(3);
    expect(imported.edges.some(e => e.sourcePort === 'b0:q0')).toBe(true);
  }, 30000);

  it('preserves Post-Norm and ReLU through import and export', () => {
    const graph = PRESETS.transformer(); graph.nodes[1].params.norm_first = 0; graph.nodes[1].params.activation = 'relu';
    const imported = roundtrip(graph), encoder = imported.nodes.find(n => n.op === 'Transformer')!;
    expect(encoder.params.norm_first).toBe(0); expect(encoder.params.activation).toBe('relu');
    expect(generatePython(imported)).toContain('norm_first=False, activation="relu"');
  }, 30000);

  it('runs sequence projections before cross-attention through validation, export and import', () => {
    const graph: Graph = {
      version: 1, name: 'Sequence projections',
      nodes: [
        { id: 'climate', name: 'Climate', op: 'Input', params: { shape: [64, 6, 101] }, position: { x: 0, y: 0 } },
        { id: 'canopy', name: 'Canopy', op: 'Input', params: { shape: [64, 6, 69] }, position: { x: 0, y: 200 } },
        { id: 'climate_projection', name: 'Climate projection', op: 'Linear', params: { out_features: 64 }, position: { x: 200, y: 0 } },
        { id: 'canopy_projection', name: 'Canopy projection', op: 'Linear', params: { out_features: 64 }, position: { x: 200, y: 200 } },
        { id: 'attention', name: 'Cross-attention', op: 'MultiHeadAttention', params: { embed_dim: 64, num_heads: 4, kv_heads: 4, attention_type: 'cross', dropout: 0 }, position: { x: 400, y: 100 } },
        { id: 'flatten', name: 'Classification pooling', op: 'Flatten', params: {}, position: { x: 600, y: 100 } },
        { id: 'head', name: 'Classification head', op: 'Linear', params: { out_features: 2 }, position: { x: 800, y: 100 } },
        { id: 'output', name: 'Output', op: 'Output', params: {}, position: { x: 1000, y: 100 } },
      ],
      edges: [
        { id: 'climate_in', source: 'climate', target: 'climate_projection' },
        { id: 'canopy_in', source: 'canopy', target: 'canopy_projection' },
        { id: 'query', source: 'climate_projection', target: 'attention', targetPort: 'query' },
        { id: 'context', source: 'canopy_projection', target: 'attention', targetPort: 'context' },
        { id: 'attention_out', source: 'attention', target: 'flatten' },
        { id: 'head_in', source: 'flatten', target: 'head' },
        { id: 'head_out', source: 'head', target: 'output' },
      ],
    };
    const analysis = analyze(graph);
    expect(analysis.valid).toBe(true);
    expect(analysis.layers.climate_projection.output).toEqual([64, 6, 64]);
    expect(analysis.layers.canopy_projection.output).toEqual([64, 6, 64]);
    const code = generatePython(graph);
    expect(code).toContain('nn.Linear(101, 64)');
    expect(code).toContain('nn.Linear(69, 64)');
    const script = `import ast, json, sys, torch
from backend.pytorch_import import import_pytorch
from backend.training import build_model
p = json.load(sys.stdin)
r = import_pytorch(p["code"])
assert r["graph"] is not None, r["diagnostics"]
backend, info = build_model(p["graph"])
imported, imported_info = build_model(r["graph"])
namespace = {"__name__": "export_test"}
exec(p["code"], namespace)
exported = namespace["VisualModel"]()
exported.load_state_dict(backend.state_dict())
for n in r["graph"]["nodes"]:
    if n["id"] in imported.layers:
        original_id = ast.literal_eval(n["name"].split("self.layers", 1)[1][1:-1])
        imported.layers[n["id"]].load_state_dict(backend.layers[original_id].state_dict())
backend.eval(); imported.eval(); exported.eval()
x = {key: torch.randn(*info["shapes"][key], requires_grad=True) for key in info["inputs"]}
y = {i["id"]: x[i["name"]] for i in r["inputs"]}
a, b, c = backend(x), exported(x), imported(y)
assert list(a.shape) == [64, 2]
assert torch.equal(a, b) and torch.equal(a, c)
assert info["totalParameters"] == imported_info["totalParameters"] == p["parameters"] == sum(v.numel() for v in exported.parameters())
b.square().sum().backward()
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in exported.parameters())
print("matched")
`;
    expect(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify({ graph, code, parameters: analysis.parameters }), encoding: 'utf8', timeout: 30000 })).toContain('matched');
  }, 30000);
});
