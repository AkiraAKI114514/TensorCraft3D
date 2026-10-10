import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze } from './analysis';
import { generatePython } from './export';
import type { Edge, Graph, Layer, SubgraphDef } from './types';

// A repeated attention used to export as a bare `nn.ModuleList` of attention layers whose forward
// loop called `instance.forward_with_ports(x, None, overrides=...)`. The importer folds a
// `nn.ModuleList` only when every element is invoked with a single positional argument, so the
// re-import died with "结构块实例需要单个输入". The export now wraps each repeated attention in a
// one-operator `nn.Module` subclass whose `forward` is the plain single-input call, so the same
// `nn.ModuleList` folding that already carried repeated non-attention ops and subgraph blocks
// carries a repeated attention too.
const SCRIPT = `import json, sys, torch
from backend.pytorch_import import import_pytorch
from backend.training import build_model
p = json.load(sys.stdin)
r = import_pytorch(p["code"])
assert r["graph"] is not None, r["diagnostics"]
groups = [n for n in r["graph"]["nodes"] if n["op"] == "Group"]
assert len(groups) == p["groups"], ("group count", len(groups), p["groups"])
if groups and p["repeat"]:
    assert groups[0].get("repeat") == p["repeat"], ("repeat", groups[0].get("repeat"), p["repeat"])
imported, info = build_model(r["graph"])
assert info["totalParameters"] == p["parameters"], (info["totalParameters"], p["parameters"])
assert sum(v.numel() for v in imported.parameters()) == p["parameters"]
if p.get("forward", True):
    namespace = {"__name__": "attention_export"}
    exec(p["code"], namespace)
    exported = namespace["VisualModel"]().eval()
    imported = imported.eval()
    left, right = list(exported.parameters()), list(imported.parameters())
    assert len(left) == len(right), ("parameter tensors", len(left), len(right))
    with torch.no_grad():
        for a, b in zip(left, right): b.copy_(a)
    order, tensors = list(p["inputs"]), [torch.randn(*shape) for shape in p["inputs"].values()]
    single = len(order) == 1
    with torch.no_grad():
        left_input = tensors[0] if single else dict(zip(order, tensors))
        right_input = tensors[0] if single else dict(zip(info["inputs"], tensors))
        assert torch.equal(exported(left_input), imported(right_input)), "folded attention export changed the computation"
print("matched")
`;

const node = (id: string, op: string, params: Layer['params'] = {}, repeat?: number): Layer =>
  ({ id, name: id, op: op as Layer['op'], params, position: { x: 0, y: 0 }, ...(repeat === undefined ? {} : { repeat }) });
const edge = (id: string, source: string, target: string): Edge => ({ id, source, target });

const SELF = { embed_dim: 8, num_heads: 2, kv_heads: 2, branches: 1, ff_dim: 16, attention_type: 'self' };

/** Input -> Transformer(self) ×repeat -> Output. */
function repeatedAttention(repeat?: number, op = 'Transformer'): Graph {
  return {
    version: 1, name: 'Repeated attention',
    nodes: [node('n0', 'Input', { shape: [2, 4, 8] }), node('n1', op, SELF, repeat), node('n2', 'Output')],
    edges: [edge('e0', 'n0', 'n1'), edge('e1', 'n1', 'n2')],
  };
}

/** Query + Context -> Transformer(cross) ×repeat -> Output. */
function crossAttention(repeat?: number): Graph {
  return {
    version: 1, name: 'Repeated cross attention',
    nodes: [node('n0', 'Input', { shape: [2, 4, 8] }), node('ctx', 'Input', { shape: [2, 6, 8] }), node('n1', 'Transformer', { ...SELF, attention_type: 'cross' }, repeat), node('n2', 'Output')],
    edges: [
      { id: 'e0', source: 'n0', target: 'n1', targetPort: 'query' },
      { id: 'e1', source: 'ctx', target: 'n1', targetPort: 'context' },
      edge('e2', 'n1', 'n2'),
    ],
  };
}

/** A block of `Transformer ×inner` -> ReLU, instantiated `outer` times. */
function nestedRepeatedAttention(outer: number, inner: number): Graph {
  const def: SubgraphDef = {
    id: 'sg1', name: 'Attention block', origin: 'manual',
    nodes: [node('a0', 'Transformer', SELF, inner), node('a1', 'ReLU')],
    edges: [edge('s0', 'a0', 'a1')],
  };
  return {
    version: 1, name: 'Nested repeated attention',
    nodes: [node('n0', 'Input', { shape: [2, 4, 8] }), { id: 'g1', name: 'g1', op: 'Group' as const, params: {}, position: { x: 0, y: 0 }, subgraph: 'sg1', repeat: outer }, node('n3', 'Linear', { out_features: 2 }), node('n4', 'Output')],
    edges: [edge('e0', 'n0', 'g1'), edge('e1', 'g1', 'n3'), edge('e2', 'n3', 'n4')],
    subgraphs: { sg1: def },
  };
}

function roundtrip(graph: Graph, groups: number, repeat: number | null, inputs: Record<string, number[]> = { n0: [2, 4, 8] }, forward = true, timeout = 60000) {
  const analysis = analyze(graph), code = generatePython(graph);
  expect(analysis.valid).toBe(true);
  const result = execFileSync('.venv/Scripts/python.exe', ['-c', SCRIPT], { input: JSON.stringify({ code, parameters: analysis.parameters, groups, repeat, inputs, forward }), encoding: 'utf8', timeout });
  return { analysis, code, result };
}

describe('repeated attention export roundtrip', () => {
  it.each([[2], [3], [6]])('reimports a self-attention folded %i times as one Group', repeat => {
    const { analysis, code, result } = roundtrip(repeatedAttention(repeat), 1, repeat);
    expect(result).toContain('matched');
    // The repeated attention becomes a reusable class in an nn.ModuleList, and the loop body is
    // the plain single-input call the importer's Group branch accepts.
    expect(code).toContain('class TensorLabRepeatedAttention_0(nn.Module):');
    expect(code.match(/TensorLabRepeatedAttention_0\(\)/g)).toHaveLength(repeat);
    expect(code).toContain('"n1": nn.ModuleList([');
    expect(code).toContain('for instance in self.layers["n1"]:');
    expect(code).toContain('values["n1"] = instance(values["n1"])');
    expect(code).not.toContain('instance.forward_with_ports');
    // Every materialised attention is independent, so the counts have to agree.
    expect(analysis.parameters).toBe(600 * repeat);
  }, 60000);

  it('folds a repeated attention past the import node budget', () => {
    // 100 attention layers would blow the 128-node import budget as a flat expansion; folding
    // them back to one repeat=100 Group is what keeps the re-import inside it.
    const { analysis, code, result } = roundtrip(repeatedAttention(100), 1, 100, { n0: [2, 4, 8] }, false);
    expect(result).toContain('matched');
    expect(code.match(/TensorLabRepeatedAttention_0\(\)/g)).toHaveLength(100);
    expect(analysis.parameters).toBe(60000);
  }, 60000);

  it('round-trips a repeated MultiHeadAttention as well', () => {
    const { analysis, result } = roundtrip(repeatedAttention(3, 'MultiHeadAttention'), 1, 3);
    expect(result).toContain('matched');
    expect(analysis.parameters).toBe(288 * 3);
  }, 60000);

  it.each([[2, 2], [3, 2], [2, 3]])('reimports a repeated attention inside a %ix%i repeated block', (outer, inner) => {
    const { analysis, code, result } = roundtrip(nestedRepeatedAttention(outer, inner), 1, outer);
    expect(result).toContain('matched');
    // The outer group folds to one instance with the outer repeat; the inner attention folds into
    // its own subgraph, which the importer nests.
    const imported = JSON.parse(execFileSync('.venv/Scripts/python.exe', ['-c', `import json,sys
from backend.pytorch_import import import_pytorch
r=import_pytorch(sys.stdin.read())
assert r['graph'] is not None, r['diagnostics']
print(json.dumps({'groups': [(n['id'], n.get('repeat')) for n in r['graph']['nodes'] if n['op']=='Group'], 'subgraphs': {k: [(n['id'], n['op'], n.get('repeat')) for n in v['nodes']] for k,v in (r['graph'].get('subgraphs') or {}).items()}}))`], { input: code, encoding: 'utf8', timeout: 60000 })) as { groups: [string, number][]; subgraphs: Record<string, [string, string, number | null][]> };
    expect(imported.groups).toHaveLength(1);
    expect(imported.groups[0][1]).toBe(outer);
    const innerGroups = Object.values(imported.subgraphs).flat().filter(([, op]) => op === 'Group');
    expect(innerGroups).toHaveLength(1);
    expect(innerGroups[0][2]).toBe(inner);
    // 600 per attention x inner x outer, plus the head Linear(8, 2).
    expect(analysis.parameters).toBe(600 * inner * outer + 18);
  }, 60000);

  it('does not wrap a repeated cross-attention, so its export still runs', () => {
    // The wrapper's `forward(self, x)` has a single slot, so routing a Query/Context pair through
    // it silently dropped the context input and made the exported module unrunnable. A repeated
    // cross-attention keeps the old forward_with_ports loop instead: it still cannot fold, but it
    // must at least still execute.
    const graph = crossAttention(2), code = generatePython(graph);
    expect(code).not.toContain('TensorLabRepeatedAttention');
    // The context input is still passed into every instance: the wrapper used to drop it.
    expect(code).toContain('values["n1"], ports["n1"] = instance.forward_with_ports(values["n1"], values["ctx"], overrides={})');
    const output = execFileSync('.venv/Scripts/python.exe', ['-c', code], { encoding: 'utf8', timeout: 60000 });
    expect(output).toContain('Output shape: (2, 4, 8)');
    // Both inputs stay wired: the context is still read, not orphaned.
    expect(code).toContain('values["ctx"] = x["ctx"]');
  }, 60000);

  it('keeps a repeat=1 attention byte-identical to a project without the field', () => {
    const withRepeat = repeatedAttention(1);
    const without = { ...repeatedAttention(1), nodes: repeatedAttention(1).nodes.map(({ repeat: _repeat, ...rest }) => rest) };
    const code = generatePython(withRepeat);
    expect(code).toBe(generatePython(without));
    // Nothing about a single attention changes: no ModuleList, no wrapper class, and the same
    // forward_with_ports call sites as before the fix. The ModelList check is scoped to the model
    // because the embedded attention helper declares `self.branches` as one.
    const model = code.slice(code.indexOf('class VisualModel'));
    expect(model).not.toContain('nn.ModuleList');
    expect(code).not.toContain('TensorLabRepeatedAttention');
    // Exact pre-change lines, so any drift in the single-instance path fails loudly.
    expect(model).toContain('            "n1": TensorLabTransformer(embed_dim=8, num_heads=2, kv_heads=2, dropout=0.1, attention_type="self", branches=1, ff_dim=16, norm_first=True, activation="gelu"),');
    expect(model).toContain('        values["n1"], ports["n1"] = self.layers["n1"].forward_with_ports(values["n0"], None, overrides={})');
  });

  it('keeps a repeat=1 MultiHeadAttention unchanged too', () => {
    const code = generatePython(repeatedAttention(1, 'MultiHeadAttention'));
    const model = code.slice(code.indexOf('class VisualModel'));
    expect(model).not.toContain('nn.ModuleList');
    expect(code).not.toContain('TensorLabRepeatedAttention');
    expect(model).toContain('"n1": TensorLabAttention(');
  });
});
