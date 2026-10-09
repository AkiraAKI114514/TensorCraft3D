import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze } from './analysis';
import { generatePython } from './export';
import { PRESETS } from './presets';
import type { Graph } from './types';

// Repeat folding is a series of N independent instances, so the exported code has
// to materialise N distinct layers and the importer has to read them back.
const SCRIPT = `import ast, json, sys, torch
from backend.pytorch_import import import_pytorch
from backend.training import build_model
p = json.load(sys.stdin)
code = p["code"]
namespace = {"__name__": "export_repeat"}
exec(code, namespace)
exported = namespace["VisualModel"]()
backend, info = build_model(p["graph"])
assert info["totalParameters"] == p["parameters"], (info["totalParameters"], p["parameters"])
exported_count = sum(v.numel() for v in exported.parameters())
assert exported_count == p["parameters"], (exported_count, p["parameters"])
# Same node ids and independent instances per node => identical state_dict layout.
exported.load_state_dict(backend.state_dict(), strict=True)
exported.eval(); backend.eval()
x = {key: torch.randn(*info["shapes"][key]) for key in info["inputs"]}
with torch.no_grad():
    if len(info["inputs"]) == 1:
        a, b = backend(x), exported(x[info["inputs"][0]])
    else:
        a, b = backend(x), exported({key: x[key] for key in info["inputs"]})
assert torch.equal(a, b), "folded export changed the computation"
r = import_pytorch(code)
assert r["graph"] is not None, r["diagnostics"]
imported, imported_info = build_model(r["graph"])
assert imported_info["totalParameters"] == p["parameters"], (imported_info["totalParameters"], p["parameters"])
assert sum(v.numel() for v in imported.parameters()) == p["parameters"]
print("matched")
`;

function foldedLinear(repeat: number): Graph {
  return {
    version: 1, name: 'Folded linear block',
    nodes: [
      { id: 'n0', name: 'Input', op: 'Input', params: { shape: [2, 4] }, position: { x: 0, y: 0 } },
      { id: 'n1', name: 'Block', op: 'Linear', params: { out_features: 4 }, position: { x: 100, y: 0 }, repeat },
      { id: 'n2', name: 'Act', op: 'ReLU', params: {}, position: { x: 200, y: 0 } },
      { id: 'n3', name: 'Head', op: 'Linear', params: { out_features: 2 }, position: { x: 300, y: 0 } },
      { id: 'n4', name: 'Output', op: 'Output', params: {}, position: { x: 400, y: 0 } },
    ],
    edges: [
      { id: 'e0', source: 'n0', target: 'n1' }, { id: 'e1', source: 'n1', target: 'n2' },
      { id: 'e2', source: 'n2', target: 'n3' }, { id: 'e3', source: 'n3', target: 'n4' },
    ],
  };
}

function foldedConv(repeat: number): Graph {
  return {
    version: 1, name: 'Folded conv block',
    nodes: [
      { id: 'n0', name: 'Input', op: 'Input', params: { shape: [2, 4, 8, 8] }, position: { x: 0, y: 0 } },
      { id: 'n1', name: 'Block', op: 'Conv2d', params: { out_channels: 4, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 100, y: 0 }, repeat },
      { id: 'n2', name: 'Output', op: 'Output', params: {}, position: { x: 200, y: 0 } },
    ],
    edges: [{ id: 'e0', source: 'n0', target: 'n1' }, { id: 'e1', source: 'n1', target: 'n2' }],
  };
}

function roundtrip(graph: Graph) {
  const analysis = analyze(graph), code = generatePython(graph);
  const result = execFileSync('.venv/Scripts/python.exe', ['-c', SCRIPT], { input: JSON.stringify({ graph, code, parameters: analysis.parameters }), encoding: 'utf8', timeout: 30000 });
  return { analysis, code, result };
}

describe('repeat folding export roundtrip', () => {
  it.each([[1], [3], [5]])('materialises an independent ModuleList for repeat=%i and reimports it', repeat => {
    const { analysis, code, result } = roundtrip(foldedLinear(repeat));
    expect(analysis.valid).toBe(true);
    expect(result).toContain('matched');
    if (repeat === 1) {
      // A single instance keeps the historical bare `nn.Linear(4, 4)` mapping.
      expect(code).not.toContain('nn.ModuleList');
      expect(code).toContain('"n1": nn.Linear(4, 4),');
    } else {
      expect(code).toContain('"n1": nn.ModuleList([');
      expect(code.match(/nn\.Linear\(4, 4\)/g)).toHaveLength(repeat);
      expect(code).toContain('for instance in self.layers["n1"]:');
    }
    // Parameters equal the analyzer's count, and the single-instance node is untouched.
    expect(analysis.parameters).toBe(20 * repeat + 10);
  });

  it('round-trips a folded Conv2d whose output shape matches its input', () => {
    const { analysis, code, result } = roundtrip(foldedConv(4));
    expect(analysis.valid).toBe(true);
    expect(analysis.parameters).toBe(148 * 4);
    expect(code.match(/nn\.Conv2d\(4, 4,/g)).toHaveLength(4);
    expect(result).toContain('matched');
  });

  it('keeps repeat=1 exports byte-identical to a project without the field', () => {
    expect(generatePython(foldedLinear(1))).toBe(generatePython({ ...foldedLinear(1), nodes: foldedLinear(1).nodes.map(({ repeat: _repeat, ...node }) => node) }));
  });

  it('round-trips an operator folded past the import node budget', () => {
    // 200 instances export as a 200-entry ModuleList. Re-importing it must fold straight back
    // to one repeat=200 node; without that the graph would need 200 nodes and be refused.
    const { analysis, code, result } = roundtrip(foldedLinear(200));
    expect(analysis.valid).toBe(true);
    expect(code.match(/nn\.Linear\(4, 4\)/g)).toHaveLength(200);
    expect(code).toContain('for instance in self.layers["n1"]:');
    expect(result).toContain('matched');
  });
});
