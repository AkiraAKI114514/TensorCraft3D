import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze } from './analysis';
import { generatePython } from './export';
import type { Edge, Graph, Layer, SubgraphDef } from './types';

// Round-tripping a grouped block is what makes a large grouped model importable at all. The
// export has to emit a reusable nn.Module class in an nn.ModuleList, because the importer only
// folds a container whose instances it can prove are identical copies: a ModuleDict never folds.
const SCRIPT = `import json, sys, torch
from backend.pytorch_import import import_pytorch
from backend.training import build_model
p = json.load(sys.stdin)
r = import_pytorch(p["code"])
assert r["graph"] is not None, r["diagnostics"]
groups = [n for n in r["graph"]["nodes"] if n["op"] == "Group"]
assert len(groups) == p["groups"], ("group count", len(groups), p["groups"])
if groups and p["repeat"]: assert groups[0].get("repeat") == p["repeat"], ("repeat", groups[0].get("repeat"))
imported, info = build_model(r["graph"])
assert info["totalParameters"] == p["parameters"], (info["totalParameters"], p["parameters"])
namespace = {"__name__": "block_export"}
exec(p["code"], namespace)
exported = namespace["VisualModel"]().eval()
imported = imported.eval()
left, right = list(exported.parameters()), list(imported.parameters())
assert len(left) == len(right), ("parameter tensors", len(left), len(right))
with torch.no_grad():
    for a, b in zip(left, right): b.copy_(a)
# The exported model keys its inputs by the original node ids while the rebuilt backend model
# keys them by the ids the importer assigned, so map both by position (the importer preserves order).
order, tensors = list(p["inputs"]), [torch.randn(*shape) for shape in p["inputs"].values()]
single = len(order) == 1
with torch.no_grad():
    left_input = tensors[0] if single else dict(zip(order, tensors))
    right_input = tensors[0] if single else dict(zip(info["inputs"], tensors))
    assert torch.equal(exported(left_input), imported(right_input)), "folded block export changed the computation"
print("matched")
`;

const node = (id: string, op: string, params: Layer['params'] = {}, subgraph?: string, repeat?: number): Layer =>
  ({ id, name: id, op: op as Layer['op'], params, position: { x: 0, y: 0 }, ...(subgraph === undefined ? {} : { subgraph }), ...(repeat === undefined ? {} : { repeat }) });
const edge = (id: string, source: string, target: string): Edge => ({ id, source, target });

/** 单实例块：Linear -> ReLU，入口出口唯一。 */
const block = (key: string, out = 4): SubgraphDef => ({
  id: key, name: 'Block', origin: 'manual',
  nodes: [node('n0', 'Linear', { out_features: out }), node('n1', 'ReLU')],
  edges: [edge('s0', 'n0', 'n1')],
});

function grouped(repeat?: number): Graph {
  return {
    version: 1, name: 'Grouped block',
    nodes: [node('n0', 'Input', { shape: [2, 4] }), node('g1', 'Group', {}, 'sg1', repeat), node('n3', 'Linear', { out_features: 2 }), node('n4', 'Output')],
    edges: [edge('e0', 'n0', 'g1'), edge('e1', 'g1', 'n3'), edge('e2', 'n3', 'n4')],
    subgraphs: { sg1: block('sg1') },
  };
}

function nested(repeat: number): Graph {
  return {
    version: 1, name: 'Nested blocks',
    nodes: [node('n0', 'Input', { shape: [2, 4] }), node('g1', 'Group', {}, 'outer', repeat), node('n3', 'Linear', { out_features: 2 }), node('n4', 'Output')],
    edges: [edge('e0', 'n0', 'g1'), edge('e1', 'g1', 'n3'), edge('e2', 'n3', 'n4')],
    subgraphs: {
      outer: { id: 'outer', name: 'Outer', origin: 'manual', nodes: [node('n0', 'Group', {}, 'inner', 3), node('n1', 'ReLU')], edges: [edge('s0', 'n0', 'n1')] },
      inner: block('inner'),
    },
  };
}

/** 同一个子图被两处实例化，但两处的输入宽度不同，所以它们不是同一个块。 */
function sharedDefinition(): Graph {
  return {
    version: 1, name: 'Shared definition',
    nodes: [node('i0', 'Input', { shape: [2, 4] }), node('i1', 'Input', { shape: [2, 8] }), node('g1', 'Group', {}, 'shared'), node('g2', 'Group', {}, 'shared'), node('c0', 'Concat', { dim: 1 }), node('o0', 'Output')],
    edges: [edge('e0', 'i0', 'g1'), edge('e1', 'i1', 'g2'), edge('e2', 'g1', 'c0'), edge('e3', 'g2', 'c0'), edge('e4', 'c0', 'o0')],
    subgraphs: { shared: block('shared') },
  };
}

function roundtrip(graph: Graph, groups: number, repeat?: number, inputs: Record<string, number[]> = { n0: [2, 4] }, timeout = 30000) {
  const analysis = analyze(graph), code = generatePython(graph);
  expect(analysis.valid).toBe(true);
  const result = execFileSync('.venv/Scripts/python.exe', ['-c', SCRIPT], { input: JSON.stringify({ code, parameters: analysis.parameters, groups, repeat: repeat ?? null, inputs }), encoding: 'utf8', timeout });
  return { analysis, code, result };
}

describe('grouped block roundtrip', () => {
  it.each([[3], [100]])('reimports a block folded %i times as one repeat instance', repeat => {
    const { analysis, code, result } = roundtrip(grouped(repeat), 1, repeat);
    expect(result).toContain('matched');
    // The block becomes a reusable class instantiated through nn.ModuleList, which is the only
    // shape the importer folds back into a single Group node.
    expect(code).toContain('class TensorLabBlock_sg1(nn.Module):');
    expect(code.match(/TensorLabBlock_sg1\(\)/g)).toHaveLength(repeat);
    expect(code).toContain('for instance in self.layers["g1"]:');
    expect(analysis.parameters).toBe(20 * repeat + 10);
  }, 60000);

  it('reimports nested blocks instead of flattening them', () => {
    const { analysis, code, result } = roundtrip(nested(2), 1, 2);
    expect(result).toContain('matched');
    // 2 outer instances x 3 inner instances, one Linear each, plus the head Linear(4, 2).
    expect(analysis.parameters).toBe(20 * 3 * 2 + 10);
    expect(code).toContain('class TensorLabBlock_outer(nn.Module):');
    expect(code).toContain('class TensorLabBlock_inner(nn.Module):');
    expect(code).toContain('TensorLabBlock_inner');
  }, 60000);

  it('falls back to the flat expansion when one definition is instantiated at two widths', () => {
    // Both groups name the same subgraph, but their inputs differ, so a single class cannot
    // serve both. The export must stay correct rather than emit one block with the wrong width.
    const { analysis, code, result } = roundtrip(sharedDefinition(), 0, undefined, { i0: [2, 4], i1: [2, 8] });
    expect(result).toContain('matched');
    expect(code).not.toContain('class TensorLabBlock_shared');
    expect(code).toContain('nn.Linear(4, 4)');
    expect(code).toContain('nn.Linear(8, 4)');
    expect(analysis.parameters).toBe(20 + 36);
  }, 60000);

  it('keeps a single-instance block runnable and matches the analyzer', () => {
    const { analysis, code, result } = roundtrip(grouped(), 0);
    expect(result).toContain('matched');
    expect(code).toContain('class TensorLabBlock_sg1(nn.Module):');
    expect(code).not.toContain('nn.ModuleList');
    expect(analysis.parameters).toBe(30);
  }, 30000);
});
