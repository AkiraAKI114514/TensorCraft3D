import { isGroup, repeatOf, type Edge, type Graph, type Layer, type SubgraphDef } from './types';

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

// `shares` 决定这一层用谁的权重，`repeat` 和结构块的定义决定实际运行哪些算子，都属于计算本身；
// 它们只在存在时写入，所以没有这些字段的旧项目 key 不变。
const computationNode = (node: Layer) => ({ id: node.id, op: node.op, params: Object.fromEntries(Object.entries(node.params).sort(([a], [b]) => a.localeCompare(b))), ...(repeatOf(node) === 1 ? {} : { repeat: node.repeat }), ...(node.subgraph === undefined ? {} : { subgraph: node.subgraph }), ...(node.shares === undefined ? {} : { shares: node.shares }) });
const computationEdge = (edge: Edge) => ({ source: edge.source, target: edge.target, ...(edge.sourcePort ? { sourcePort: edge.sourcePort } : {}), ...(edge.targetPort ? { targetPort: edge.targetPort } : {}) });

// Names, graph title, edge IDs and positions do not change the computation.
export function computationKey(graph: Graph): string {
  // Only the block definitions the graph actually runs count, nested ones included.
  const definitions: Record<string, SubgraphDef> = graph.subgraphs ?? {}, used = new Set<string>();
  const pending = graph.nodes.filter(isGroup).map(node => node.subgraph);
  while (pending.length) {
    const name = pending.pop();
    if (name === undefined || used.has(name) || !Object.hasOwn(definitions, name)) continue;
    used.add(name);
    pending.push(...definitions[name].nodes.filter(isGroup).map(node => node.subgraph));
  }
  return JSON.stringify({
    nodes: graph.nodes.map(computationNode),
    edges: graph.edges.map(computationEdge),
    ...(used.size ? { subgraphs: Object.fromEntries([...used].sort().map(name => [name, { nodes: definitions[name].nodes.map(computationNode), edges: definitions[name].edges.map(computationEdge) }])) } : {})
  });
}

export function matchingTrainedModel(graph: Graph, result: TrainedModelResult | null): TrainedModelMetadata | null {
  return result?.graphKey === computationKey(graph) ? result.metadata : null;
}
