import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze, validateGraph } from './analysis';
import { generatePython } from './export';
import { autoGroup, manualGroup } from './grouping';
import { computationKey } from './trainedModel';
import type { Graph, Layer } from './types';

const layer = (id: string, op: Layer['op'], params: Layer['params'] = {}, extra: Partial<Layer> = {}): Layer => ({ id, name: id, op, params, position: { x: 0, y: 0 }, ...extra });

/** Input -> cell -> act -> again(shares cell) -> act2 -> head -> Output：一组权重被调用两次。 */
function tied(shares: string | null = 'cell'): Graph {
  const ids = ['in', 'cell', 'act', 'again', 'act2', 'head', 'out'];
  return {
    version: 1, name: 'Tied cell',
    nodes: [
      layer('in', 'Input', { shape: [2, 8] }), layer('cell', 'Linear', { out_features: 8 }), layer('act', 'Tanh'),
      layer('again', 'Linear', { out_features: 8 }, shares === null ? {} : { shares }), layer('act2', 'Tanh'),
      layer('head', 'Linear', { out_features: 2 }), layer('out', 'Output'),
    ],
    edges: ids.slice(1).map((target, index) => ({ id: `e${index}`, source: ids[index], target })),
  };
}

describe('weight sharing', () => {
  it('counts a shared layer\'s weights once and keeps its shapes', () => {
    const shared = analyze(tied()), separate = analyze(tied(null));
    expect(shared.valid, JSON.stringify(shared.diagnostics)).toBe(true);
    expect(separate.parameters).toBe(72 + 72 + 18);
    expect(shared.parameters).toBe(72 + 18);
    expect(shared.layers.again.parameters).toBe(0);
    expect(shared.layers.cell.parameters).toBe(72);
    expect(shared.layers.again.output).toEqual([2, 8]);
  });

  it('names the reason a reference cannot be shared', () => {
    const reason = (graph: Graph) => analyze(graph).diagnostics.filter(d => d.code === 'SHARED_WEIGHTS').map(d => `${d.nodeId}: ${d.message}`);
    expect(reason(tied('ghost'))).toEqual(['again: 共享权重需要指向另一层']);
    expect(reason(tied('again'))).toEqual(['again: 共享权重需要指向另一层']);
    expect(reason(tied('head'))).toEqual(['again: 共享权重的两层必须是相同算子且参数一致']);
    expect(reason(tied('act'))).toEqual(['again: 共享权重的两层必须是相同算子且参数一致']);
    const folded = tied(); folded.nodes[1].repeat = 2;
    expect(reason(folded)).toEqual(['again: 共享权重的层不能使用 repeat 折叠']);
    const chained = tied(); chained.nodes[5] = layer('head', 'Linear', { out_features: 8 }, { shares: 'again' });
    expect(reason(chained)).toEqual(['head: 共享权重只能指向拥有参数的层，不能指向另一个共享层']);
    const free = tied(); free.nodes[4] = layer('act2', 'Tanh', {}, { shares: 'act' });
    expect(reason(free)).toEqual(['act2: 被共享的层没有可共享的参数']);
    for (const graph of [tied('ghost'), folded, chained, free]) { expect(analyze(graph).valid).toBe(false); expect(() => generatePython(graph)).toThrow(); }
  });

  it('only admits sharing between top-level layers in a project file', () => {
    expect(validateGraph(tied())).toEqual(tied());
    const numeric = tied(); (numeric.nodes[3] as unknown as { shares: unknown }).shares = 3;
    expect(() => validateGraph(numeric)).toThrow('共享权重引用无效');
    const group: Graph = { ...tied(null), subgraphs: { block: { id: 'block', name: 'Block', origin: 'manual', nodes: [layer('n0', 'Linear', { out_features: 8 }), layer('n1', 'Linear', { out_features: 8 }, { shares: 'n0' })], edges: [{ id: 's0', source: 'n0', target: 'n1' }] } } };
    expect(() => validateGraph(group)).toThrow('内的层不能共享权重');
  });

  it('is part of the computation a trained snapshot is bound to, without moving existing keys', () => {
    const before = tied(null);
    // 没有共享的图，key 里不出现这个字段：已有项目的训练快照绑定不受影响。
    expect(computationKey(before)).not.toContain('shares');
    expect(computationKey(tied())).not.toBe(computationKey(before));
    expect(computationKey(tied())).toContain('"shares":"cell"');
  });

  it('is never folded into a block, automatically or by hand', () => {
    // 四段同构的 Linear→Tanh，其中后三段共享第一段的权重：签名完全相同，正是自动分组会折叠的形状。
    const ids = ['in', ...Array.from({ length: 4 }, (_, i) => [`fc${i}`, `act${i}`]).flat(), 'out'];
    const stack = (shared: boolean): Graph => ({
      version: 1, name: 'Tied stack',
      nodes: ids.map(id => id === 'in' ? layer(id, 'Input', { shape: [2, 8] }) : id === 'out' ? layer(id, 'Output') : id.startsWith('fc') ? layer(id, 'Linear', { out_features: 8 }, shared && id !== 'fc0' ? { shares: 'fc0' } : {}) : layer(id, 'Tanh')),
      edges: ids.slice(1).map((target, index) => ({ id: `e${index}`, source: ids[index], target })),
    });
    expect(autoGroup(stack(false)).groups).toBe(1);
    const tiedStack = stack(true), result = autoGroup(tiedStack);
    expect(result.groups).toBe(0);
    expect(result.graph).toBe(tiedStack);
    expect(analyze(tiedStack).parameters).toBe(72);
    expect(manualGroup(tiedStack, ['fc1', 'act1'])).toEqual({ error: '共享权重的层不能放进结构块' });
    // 被共享的那一层同样留在顶层。
    expect(manualGroup(tiedStack, ['fc0', 'act0'])).toEqual({ error: '共享权重的层不能放进结构块' });
    expect('error' in manualGroup(stack(false), ['fc1', 'act1'])).toBe(false);
  });

  it('exports one module called twice and re-imports it as the same shared computation', () => {
    const graph = tied(), analysis = analyze(graph), code = generatePython(graph);
    // 共享层不声明自己的模块，前向里直接调用被共享的那一个。
    expect(code).not.toContain('"again": nn.Linear');
    expect(code).toContain('values["again"] = self.layers["cell"](values["act"])');
    const script = `import ast, json, sys, torch
from backend.pytorch_import import import_pytorch
from backend.training import build_model
p = json.load(sys.stdin)
r = import_pytorch(p["code"])
assert r["graph"] is not None, r["diagnostics"]
original, info = build_model(p["graph"])
imported, imported_info = build_model(r["graph"])
namespace = {"__name__": "shared_export"}
exec(p["code"], namespace)
exported = namespace["VisualModel"]()
assert info["totalParameters"] == imported_info["totalParameters"] == p["parameters"] == sum(v.numel() for v in exported.parameters()), "parameter counts"
exported.load_state_dict(original.state_dict())
for node in r["graph"]["nodes"]:
    if node["id"] in imported.layers:
        source = ast.literal_eval(node["name"].split("self.layers", 1)[1][1:-1])
        imported.layers[node["id"]].load_state_dict(original.layers[source].state_dict())
x = torch.randn(2, 8)
a, b, c = original.eval()(x), exported.eval()(x), imported.eval()(x)
assert torch.equal(a, b) and torch.equal(a, c), "sharing changed the computation"
print(json.dumps([n.get("shares") for n in r["graph"]["nodes"] if n["op"] == "Linear"]))
`;
    const output = execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify({ graph, code, parameters: analysis.parameters }), encoding: 'utf8', timeout: 60000 });
    const shares = JSON.parse(output.trim().split('\n').at(-1)!) as (string | null)[];
    // 三个 Linear：被共享的、共享它的、独立的分类头。
    expect(shares.filter(value => value !== null)).toHaveLength(1);
    expect(shares[0]).toBeNull();
    expect(shares[1]).not.toBeNull();
    expect(shares[2]).toBeNull();
  }, 60000);
});
