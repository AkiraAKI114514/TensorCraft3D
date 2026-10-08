import type { Graph } from './types';

export interface TrainedModelMetadata {
  modelId: string;
  trainingRunId: string;
  graphFingerprint: string;
  createdAt: string;
  device: 'CPU' | 'CUDA';
  dataset: 'synthetic' | 'csv';
  seed: number;
  epochsCompleted: number;
  weightsEpoch: number;
  reason: 'completed' | 'early_stopping';
  preprocessing: 'none' | 'csv-standardized';
  storage: 'backend-memory';
}

export interface TrainedModelResult { graphKey: string; metadata: TrainedModelMetadata; }

// Names, graph title, edge IDs and positions do not change the computation.
export function computationKey(graph: Graph): string {
  return JSON.stringify({
    nodes: graph.nodes.map(node => ({ id: node.id, op: node.op, params: Object.fromEntries(Object.entries(node.params).sort(([a], [b]) => a.localeCompare(b))) })),
    edges: graph.edges.map(edge => ({ source: edge.source, target: edge.target, ...(edge.sourcePort ? { sourcePort: edge.sourcePort } : {}), ...(edge.targetPort ? { targetPort: edge.targetPort } : {}) }))
  });
}

export function matchingTrainedModel(graph: Graph, result: TrainedModelResult | null): TrainedModelMetadata | null {
  return result?.graphKey === computationKey(graph) ? result.metadata : null;
}
