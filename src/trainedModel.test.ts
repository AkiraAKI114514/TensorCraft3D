import { describe, expect, it } from 'vitest';
import { PRESETS } from './presets';
import { computationKey, matchingTrainedModel, type TrainedModelMetadata } from './trainedModel';
import type { Graph, Layer } from './types';

export const modelMetadata: TrainedModelMetadata = {
  modelId: 'model-fixture', trainingRunId: 'training-fixture', graphFingerprint: 'fingerprint', createdAt: '2026-10-08T00:00:00Z',
  device: 'CPU', dataset: 'synthetic', seed: 42, epochsCompleted: 4, weightsEpoch: 1, reason: 'early_stopping', preprocessing: 'none', storage: 'backend-memory'
};

describe('trained model graph binding', () => {
  it('keeps layout and naming changes bound to the same computation', () => {
    const graph = PRESETS.mlp(), key = computationKey(graph), result = { graphKey: key, metadata: modelMetadata };
    const renamed = { ...graph, name: 'Renamed', nodes: graph.nodes.map(node => ({ ...node, name: 'Layer', position: { x: 7, y: 9 } })), edges: graph.edges.map(edge => ({ ...edge, id: 'renamed-' + edge.id })) };
    expect(computationKey(renamed)).toBe(key);
    expect(matchingTrainedModel(renamed, result)).toBe(modelMetadata);
  });

  it('invalidates parameters, input shapes and projection wiring, and supports restoring the graph', () => {
    const graph = PRESETS.cross_attention(), result = { graphKey: computationKey(graph), metadata: modelMetadata };
    for (const changed of [
      { ...graph, nodes: graph.nodes.map((node, i) => i === 0 ? { ...node, params: { ...node.params, shape: [1, 2, 64] } } : node) },
      { ...graph, nodes: graph.nodes.map((node, i) => i === 1 ? { ...node, params: { ...node.params, dropout: 0.75 } } : node) },
      { ...graph, edges: graph.edges.map((edge, i) => i === 0 ? { ...edge, targetPort: 'b0:q0' } : edge) }
    ]) expect(matchingTrainedModel(changed, result)).toBeNull();
    expect(matchingTrainedModel(structuredClone(graph), result)).toBe(modelMetadata);
    expect(matchingTrainedModel(graph, null)).toBeNull();
  });

  it('canonicalizes parameter key order but preserves execution edge order', () => {
    const graph = PRESETS.cnn(), reversed = { ...graph, nodes: graph.nodes.map(node => ({ ...node, params: Object.fromEntries(Object.entries(node.params).reverse()) })) };
    expect(computationKey(reversed)).toBe(computationKey(graph));
    expect(computationKey({ ...graph, edges: [...graph.edges].reverse() })).not.toBe(computationKey(graph));
  });

  it('keeps the key of a graph without folded repeats or blocks', () => {
    const key = JSON.parse(computationKey(PRESETS.cnn())) as { nodes: Record<string, unknown>[] };
    expect(Object.keys(key)).toEqual(['nodes', 'edges']);
    for (const node of key.nodes) expect(Object.keys(node)).toEqual(['id', 'op', 'params']);
  });

  it('covers folded repeats and the contents of every block the graph runs', () => {
    const layer = (id: string, op: Layer['op'], params: Layer['params'] = {}, extra: Partial<Layer> = {}): Layer => ({ id, name: id, op, params, position: { x: 0, y: 0 }, ...extra });
    const blocks = (): Graph => ({
      version: 1, name: 'Blocks',
      nodes: [layer('in', 'Input', { shape: [2, 8] }), layer('stack', 'Group', {}, { subgraph: 'cell', repeat: 2 }), layer('head', 'Linear', { out_features: 2 }), layer('out', 'Output')],
      edges: [{ id: 'e0', source: 'in', target: 'stack' }, { id: 'e1', source: 'stack', target: 'head' }, { id: 'e2', source: 'head', target: 'out' }],
      subgraphs: {
        cell: { id: 'cell', name: 'Cell', origin: 'manual', nodes: [layer('fc', 'Linear', { out_features: 8 }), layer('tail', 'Group', {}, { subgraph: 'deep' })], edges: [{ id: 'c0', source: 'fc', target: 'tail' }] },
        deep: { id: 'deep', name: 'Deep', origin: 'manual', nodes: [layer('act', 'Tanh')], edges: [] },
        unused: { id: 'unused', name: 'Unused', origin: 'manual', nodes: [layer('x', 'ReLU')], edges: [] },
      },
    });
    const key = computationKey(blocks());
    // Names and positions inside a block, an explicit repeat of 1 and a definition nothing runs
    // do not change what the graph computes.
    const same = blocks();
    same.subgraphs!.cell = { ...same.subgraphs!.cell, name: 'Renamed', nodes: same.subgraphs!.cell.nodes.map(node => ({ ...node, name: 'shown', position: { x: 9, y: 9 } })) };
    same.nodes[2].repeat = 1;
    delete same.subgraphs!.unused;
    expect(computationKey(same)).toBe(key);
    for (const edit of [
      (graph: Graph) => { graph.nodes[1].repeat = 3; },
      (graph: Graph) => { graph.nodes[2].repeat = 2; },
      (graph: Graph) => { graph.subgraphs!.cell.nodes[0].params.out_features = 4; },
      (graph: Graph) => { graph.subgraphs!.cell.nodes[1].repeat = 2; },
      (graph: Graph) => { graph.subgraphs!.deep.nodes[0].op = 'ReLU'; },
    ]) {
      const changed = blocks(); edit(changed);
      expect(computationKey(changed)).not.toBe(key);
    }
  });
});
