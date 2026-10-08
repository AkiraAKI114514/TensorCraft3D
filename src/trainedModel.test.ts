import { describe, expect, it } from 'vitest';
import { PRESETS } from './presets';
import { computationKey, matchingTrainedModel, type TrainedModelMetadata } from './trainedModel';

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
});
