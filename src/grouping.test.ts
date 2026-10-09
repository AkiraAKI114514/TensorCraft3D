import { describe, expect, it } from 'vitest';
import { analyze } from './analysis';
import { autoGroup, dissolveAutoGroups, dissolveGroup, manualGroup } from './grouping';
import type { Edge, Graph, Layer } from './types';

const node = (id: string, op: Layer['op'], params: Layer['params'] = {}): Layer => ({ id, name: id, op, params, position: { x: 0, y: 0 } });
const edge = (source: string, target: string): Edge => ({ id: `${source}->${target}`, source, target });

/** 一个块重复 N 次，每块是 Linear(保形状) + ReLU。 */
function repeatedBlocks(repeat: number, width = 4): Graph {
  const nodes = [node('i', 'Input', { shape: [2, width] }), node('o', 'Output')];
  const edges: Edge[] = [];
  let previous = 'i';
  for (let index = 0; index < repeat; index++) {
    nodes.push(node(`l${index}`, 'Linear', { out_features: width }), node(`a${index}`, 'ReLU'));
    edges.push(edge(previous, `l${index}`), edge(`l${index}`, `a${index}`));
    previous = `a${index}`;
  }
  edges.push(edge(previous, 'o'));
  return { version: 1, name: 'repeated', nodes, edges };
}

describe('autoGroup', () => {
  it('folds a repeated block into one instance and keeps accounting identical', () => {
    const flat = repeatedBlocks(4);
    const before = analyze(flat);
    const { graph, groups } = autoGroup(flat);
    expect(groups).toBe(1);
    const after = analyze(graph);
    expect(after.valid).toBe(true);
    expect(after.parameters).toBe(before.parameters);
    expect(after.activationBytes).toBe(before.activationBytes);
  });

  it('collapses the node count to the blocks plus the terminals', () => {
    const { graph } = autoGroup(repeatedBlocks(5));
    // Input + Output + 一个块实例
    expect(graph.nodes.length).toBe(3);
    expect(graph.nodes.filter(n => n.op === 'Group')).toHaveLength(1);
    expect(graph.nodes.find(n => n.op === 'Group')!.repeat).toBe(5);
  });

  it('leaves a graph with no repetition alone', () => {
    const graph: Graph = { version: 1, name: 'chain', nodes: [node('i', 'Input', { shape: [2, 4] }), node('l', 'Linear', { out_features: 8 }), node('o', 'Output')], edges: [edge('i', 'l'), edge('l', 'o')] };
    expect(autoGroup(graph).groups).toBe(0);
  });

  it('does not fold a block whose internals are wired differently', () => {
    // 同样的算子序列，但第二段把 ReLU 接在 Linear 之前，拓扑不同构。
    const nodes = [node('i', 'Input', { shape: [2, 4] }), node('o', 'Output'),
      node('l0', 'Linear', { out_features: 4 }), node('a0', 'ReLU'),
      node('l1', 'Linear', { out_features: 4 }), node('a1', 'ReLU')];
    const edges = [edge('i', 'l0'), edge('l0', 'a0'), edge('a0', 'a1'), edge('a1', 'l1'), edge('l1', 'o')];
    const graph: Graph = { version: 1, name: 'twisted', nodes, edges };
    const result = autoGroup(graph);
    expect(analyze(result.graph).parameters).toBe(analyze(graph).parameters);
  });

  it('never folds a block that has an extra input landing mid-block', () => {
    // 块外直接插进块中间（s 接到 add0 / add1，而不是块入口 l0）：折叠会丢掉这条接线。
    const nodes = [node('i', 'Input', { shape: [2, 4] }), node('s', 'Input', { shape: [2, 4] }), node('o', 'Output'),
      node('l0', 'Linear', { out_features: 4 }), node('add0', 'Add'),
      node('l1', 'Linear', { out_features: 4 }), node('add1', 'Add')];
    const edges = [edge('i', 'l0'), edge('l0', 'add0'), edge('add0', 'l1'), edge('l1', 'add1'), edge('add1', 'o'), edge('s', 'add0'), edge('s', 'add1')];
    const graph: Graph = { version: 1, name: 'sideway', nodes, edges };
    const result = autoGroup(graph);
    expect(result.groups).toBe(0);
    expect(analyze(result.graph).parameters).toBe(analyze(graph).parameters);
  });

  it('is idempotent: an already-grouped graph is left alone', () => {
    const { graph } = autoGroup(repeatedBlocks(3));
    expect(autoGroup(graph).groups).toBe(0);
  });

  it('produces graphs that still analyze cleanly through expansion', () => {
    const { graph } = autoGroup(repeatedBlocks(6, 8));
    const expanded = analyze(graph);
    expect(expanded.valid).toBe(true);
    expect(expanded.diagnostics.filter(d => d.level === 'error')).toHaveLength(0);
  });
});

describe('manualGroup', () => {
  const chain = (): Graph => ({ version: 1, name: 'chain', nodes: [node('i', 'Input', { shape: [2, 4] }), node('l', 'Linear', { out_features: 4 }), node('a', 'ReLU'), node('o', 'Output')], edges: [edge('i', 'l'), edge('l', 'a'), edge('a', 'o')] });

  it('folds a hand-picked run and keeps accounting identical', () => {
    const graph = chain();
    const result = manualGroup(graph, ['l', 'a'], 'Hand Block');
    expect('graph' in result).toBe(true);
    const grouped = (result as { graph: Graph }).graph;
    expect(grouped.nodes.filter(n => n.op === 'Group')).toHaveLength(1);
    expect(grouped.subgraphs!.manual_1.origin).toBe('manual');
    expect(grouped.subgraphs!.manual_1.name).toBe('Hand Block');
    expect(analyze(grouped).parameters).toBe(analyze(graph).parameters);
  });

  it('refuses a selection that is not a single-entry single-exit run', () => {
    // 只选 l：它是入口，但 a 不在块内，于是出边和入边各自唯一——这里选 l 单独一个应当被拒绝（不足两个节点）。
    const graph = chain();
    expect(manualGroup(graph, ['l'])).toHaveProperty('error');
  });

  it('refuses a selection whose outside edges land on more than one node', () => {
    // i→l0 与 s→add 是两条来自块外的边，落在块内不同节点上：块会有两个入口，必须拒绝。
    const graph: Graph = { version: 1, name: 'two-in', nodes: [node('i', 'Input', { shape: [2, 4] }), node('s', 'Input', { shape: [2, 4] }), node('l0', 'Linear', { out_features: 4 }), node('add', 'Add'), node('o', 'Output')], edges: [edge('i', 'l0'), edge('l0', 'add'), edge('s', 'add'), edge('add', 'o')] };
    expect(manualGroup(graph, ['l0', 'add'])).toHaveProperty('error');
  });

  it('round-trips: manual group then dissolve restores the accounting', () => {
    const graph = chain();
    const grouped = (manualGroup(graph, ['l', 'a']) as { graph: Graph; id: string });
    const dissolved = dissolveGroup(grouped.graph, grouped.id);
    expect('graph' in dissolved).toBe(true);
    const restored = (dissolved as { graph: Graph }).graph;
    expect(restored.nodes.some(n => n.op === 'Group')).toBe(false);
    expect(analyze(restored).parameters).toBe(analyze(graph).parameters);
    expect(restored.edges).toHaveLength(graph.edges.length);
  });

  it('rejects dissolving a node that is not a block', () => {
    expect(dissolveGroup(chain(), 'l')).toHaveProperty('error');
  });
});

describe('dissolveAutoGroups', () => {
  it('round-trips an auto-grouped graph back to the flat accounting', () => {
    const flat = repeatedBlocks(4);
    const { graph } = autoGroup(flat);
    const { graph: restored, groups } = dissolveAutoGroups(graph);
    expect(groups).toBe(1);
    expect(restored.nodes.some(n => n.op === 'Group')).toBe(false);
    expect(analyze(restored).parameters).toBe(analyze(flat).parameters);
  });

  it('leaves manual groups untouched', () => {
    const manual: Graph = {
      version: 1, name: 'manual',
      nodes: [node('i', 'Input', { shape: [2, 4] }), { ...node('g', 'Group'), subgraph: 'hand', repeat: 2 }, node('o', 'Output')],
      edges: [edge('i', 'g'), edge('g', 'o')],
      subgraphs: { hand: { id: 'hand', name: 'hand', origin: 'manual', nodes: [node('l', 'Linear', { out_features: 4 })], edges: [] } },
    };
    const { groups } = dissolveAutoGroups(manual);
    expect(groups).toBe(0);
  });
});
