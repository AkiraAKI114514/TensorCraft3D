import { describe, expect, it } from 'vitest';
import { analyze, validateGraph } from './analysis';
import { expandGraph, topLevelId, SubgraphError } from './subgraph';
import type { Edge, Graph, Layer, SubgraphDef } from './types';

const node = (id: string, op: Layer['op'], params: Layer['params'] = {}, extra: Partial<Layer> = {}): Layer =>
  ({ id, name: id, op, params, position: { x: 0, y: 0 }, ...extra });
const edge = (source: string, target: string, id = `${source}->${target}`): Edge => ({ id, source, target });

/** 展平写法的 3 个同构块。 */
function flat(): Graph {
  const nodes = [node('i', 'Input', { shape: [2, 4] }), node('o', 'Output')];
  const edges: Edge[] = [];
  let previous = 'i';
  for (let index = 0; index < 3; index++) {
    nodes.push(node(`l${index}`, 'Linear', { out_features: 4 }), node(`a${index}`, 'ReLU'));
    edges.push(edge(previous, `l${index}`), edge(`l${index}`, `a${index}`));
    previous = `a${index}`;
  }
  edges.push(edge(previous, 'o'));
  return { version: 1, name: 'flat', nodes, edges };
}

/** 同一个模型的子图写法：一个块实例 ×3。 */
function grouped(repeat = 3): Graph {
  const block: SubgraphDef = {
    id: 'block', name: 'Linear + ReLU', origin: 'auto',
    nodes: [node('l', 'Linear', { out_features: 4 }), node('a', 'ReLU')],
    edges: [edge('l', 'a')],
  };
  return {
    version: 1, name: 'grouped',
    nodes: [node('i', 'Input', { shape: [2, 4] }), node('g', 'Group', {}, { subgraph: 'block', repeat }), node('o', 'Output')],
    edges: [edge('i', 'g'), edge('g', 'o')],
    subgraphs: { block },
  };
}

describe('subgraph expansion', () => {
  it('is semantically lossless: grouping changes no accounting', () => {
    const before = analyze(flat()), after = analyze(grouped());
    expect(after.valid).toBe(true);
    expect(after.parameters).toBe(before.parameters);
    expect(after.activationBytes).toBe(before.activationBytes);
  });

  it('collapses results back onto the top-level node', () => {
    const groupedNodes = analyze(grouped());
    // 顶层只有 Input / Group / Output 三个节点参与结果聚合。
    expect(Object.keys(groupedNodes.layers).sort()).toEqual(['g', 'i', 'o']);
    // 参数全部记在那个块实例上，等于展开后的三个块之和。
    const flatNodes = analyze(flat());
    const flatTotal = flatNodes.layers.l0.parameters + flatNodes.layers.l1.parameters + flatNodes.layers.l2.parameters;
    expect(groupedNodes.layers.g.parameters).toBe(flatTotal);
    // 块的输出形状等于最后一个内部节点的输出。
    expect(groupedNodes.layers.g.output).toEqual(flatNodes.layers.a2.output);
  });

  it('expands to a graph with no Group node left', () => {
    const expanded = expandGraph(grouped(2));
    expect(expanded.nodes.length).toBe(2 + 2 * 2);
    expect(expanded.nodes.some(n => n.op === 'Group')).toBe(false);
    expect(expanded.subgraphs).toBeUndefined();
  });

  it('routes expanded ids back to their top-level owner', () => {
    const expanded = expandGraph(grouped(2));
    for (const id of expanded.nodes.map(n => n.id)) expect(['i', 'g', 'o']).toContain(topLevelId(id));
  });

  it('supports nested subgraphs', () => {
    const inner: SubgraphDef = { id: 'inner', name: 'inner', origin: 'manual', nodes: [node('l', 'Linear', { out_features: 4 })], edges: [] };
    const outer: SubgraphDef = { id: 'outer', name: 'outer', origin: 'manual', nodes: [node('x', 'Group', {}, { subgraph: 'inner' }), node('a', 'ReLU')], edges: [edge('x', 'a')] };
    const nested: Graph = {
      version: 1, name: 'nested',
      nodes: [node('i', 'Input', { shape: [2, 4] }), node('g', 'Group', {}, { subgraph: 'outer', repeat: 2 }), node('o', 'Output')],
      edges: [edge('i', 'g'), edge('g', 'o')],
      subgraphs: { inner, outer },
    };
    const expanded = analyze(nested);
    expect(expanded.valid).toBe(true);
    // 两层展开：2 个外层实例 × 1 个内层实例 × 1 个 Linear。
    expect(expanded.layers.g.parameters).toBe(2 * (4 * 4 + 4));
  });

  it('rejects a subgraph without a unique entry or exit', () => {
    const twoEntries: SubgraphDef = { id: 'bad', name: 'bad', origin: 'auto', nodes: [node('a', 'ReLU'), node('b', 'ReLU')], edges: [] };
    const graph: Graph = { ...grouped(), nodes: [node('i', 'Input', { shape: [2, 4] }), node('g', 'Group', {}, { subgraph: 'bad' }), node('o', 'Output')], edges: [edge('i', 'g'), edge('g', 'o')], subgraphs: { bad: twoEntries } };
    const result = analyze(graph);
    expect(result.valid).toBe(false);
    expect(result.diagnostics[0].code).toBe('SUBGRAPH');
  });

  it('reports a missing subgraph reference rather than throwing', () => {
    const graph: Graph = { ...grouped(), nodes: [node('i', 'Input', { shape: [2, 4] }), node('g', 'Group', {}, { subgraph: 'ghost' }), node('o', 'Output')] };
    const result = analyze(graph);
    expect(result.valid).toBe(false);
    expect(result.diagnostics[0].message).toContain('ghost');
  });

  it('keeps the flat path byte-identical when no subgraphs are present', () => {
    const graph = flat();
    expect(expandGraph(graph)).toBe(graph);
  });
});

describe('validateGraph subgraph rules', () => {
  const valid = grouped();

  it('accepts a well-formed grouped project', () => {
    expect(validateGraph(JSON.parse(JSON.stringify(valid)))).toBeTruthy();
  });

  it('requires a Group node to name a subgraph', () => {
    const broken = JSON.parse(JSON.stringify(valid)) as Graph;
    delete broken.nodes.find(n => n.op === 'Group')!.subgraph;
    expect(() => validateGraph(broken)).toThrow();
  });

  it('rejects subgraph on a non-Group node', () => {
    const broken = JSON.parse(JSON.stringify(valid)) as Graph;
    broken.nodes[0].subgraph = 'block';
    expect(() => validateGraph(broken)).toThrow();
  });

  it('rejects a dangling subgraph reference', () => {
    const broken = JSON.parse(JSON.stringify(valid)) as Graph;
    broken.nodes.find(n => n.op === 'Group')!.subgraph = 'missing';
    expect(() => validateGraph(broken)).toThrow();
  });

  it('rejects a cyclic subgraph reference', () => {
    const broken = JSON.parse(JSON.stringify(valid)) as Graph;
    broken.subgraphs!.block.nodes.push(node('self', 'Group', {}, { subgraph: 'block' }));
    expect(() => validateGraph(broken)).toThrow();
  });

  it('accepts a project with no subgraphs at all', () => {
    expect(validateGraph(JSON.parse(JSON.stringify(flat())))).toBeTruthy();
  });
});

describe('expandGraph guards', () => {
  it('throws a typed error for a missing definition', () => {
    const graph: Graph = { ...grouped(), nodes: [node('i', 'Input', { shape: [2, 4] }), node('g', 'Group', {}, { subgraph: 'ghost' }), node('o', 'Output')] };
    expect(() => expandGraph(graph)).toThrow(SubgraphError);
  });
});
