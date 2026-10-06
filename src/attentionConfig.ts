import type { Edge, Graph, Layer, Params } from './types';

export const ATTENTION_TYPES = {
  self: 'Self-Attention',
  multi_query: 'MQA · Multi-Query',
  grouped_query: 'GQA · Grouped-Query',
  cross: 'Cross-Attention',
  multi_branch: 'Multi-Branch'
} as const;
export type AttentionType = keyof typeof ATTENTION_TYPES;
export const isAttention = (op: string) => op === 'MultiHeadAttention' || op === 'Transformer';
export const isCrossAttention = (node: Layer) => isAttention(node.op) && node.params.attention_type === 'cross';
// Legacy projects omit these parameters. Self/cross default to one KV head per Q head.
export function attentionConfig(p: Params) {
  const type = (p.attention_type ?? 'self') as AttentionType;
  const heads = Number(p.num_heads ?? 1);
  const kvHeads = Number(p.kv_heads ?? (type === 'multi_query' ? 1 : heads));
  return { type, heads, kvHeads, branches: Number(p.branches ?? 1) };
}

export function incomingEdges(graph: Graph, node: Layer) {
  const edges = graph.edges.filter(e => e.target === node.id);
  const base = edges.filter(e => !isProjectionPort(e.targetPort));
  const resolved = isCrossAttention(node) ? base.map((edge, i) => ({ edge, role: edge.targetPort ?? (i === 0 ? 'query' : 'context') })).sort((a, b) => Number(a.role === 'context') - Number(b.role === 'context')).map(item => item.edge) : base;
  return [...resolved, ...edges.filter(e => isProjectionPort(e.targetPort))];
}

export function crossInputRole(graph: Graph, node: Layer, edgeId: string) {
  const edges = graph.edges.filter(e => e.target === node.id && !isProjectionPort(e.targetPort));
  const index = edges.findIndex(e => e.id === edgeId);
  return edges[index]?.targetPort ?? (index === 0 ? 'query' : 'context');
}

export const isProjectionPort = (port?: string) => Boolean(port && /^b\d+:[qkv]\d+$/.test(port));
export interface ProjectionPort { id: string; branch: number; index: number; role: 'Q' | 'K' | 'V'; label: string; }
export function projectionPorts(node: Layer): ProjectionPort[] {
  if (!isAttention(node.op)) return [];
  const cfg = attentionConfig(node.params);
  const heads = Math.max(1, Math.min(16, Math.floor(cfg.heads) || 1)), kv = Math.max(1, Math.min(heads, Math.floor(cfg.kvHeads) || 1));
  return Array.from({ length: Math.max(1, Math.min(8, Math.floor(cfg.branches) || 1)) }, (_, branch) =>
    (['Q', 'K', 'V'] as const).flatMap(role => Array.from({ length: role === 'Q' ? heads : kv }, (_, index) => ({
      id: `b${branch}:${role.toLowerCase()}${index}`, branch, index, role,
      label: `${cfg.branches > 1 ? `B${branch + 1} · ` : ''}${role}${index + 1}`
    })))
  ).flat();
}
export const projectionLabel = (node: Layer, port?: string) => projectionPorts(node).find(p => p.id === port)?.label ?? port ?? 'Output';
export const edgeKey = (edge: Pick<Edge, 'source' | 'target' | 'sourcePort' | 'targetPort'>) => JSON.stringify([edge.source, edge.sourcePort ?? '', edge.target, edge.targetPort ?? '']);
export function edgeOutputShape(graph: Graph, edge: Edge, shapes: Record<string, { output: number[] }>): number[] | undefined {
  const shape = shapes[edge.source]?.output;
  if (!shape || !edge.sourcePort) return shape;
  const source = graph.nodes.find(n => n.id === edge.source)!;
  const port = projectionPorts(source).find(p => p.id === edge.sourcePort);
  if (!port) return undefined;
  const cfg = attentionConfig(source.params), base = incomingEdges(graph, source).filter(e => !isProjectionPort(e.targetPort));
  const override = graph.edges.find(e => e.target === source.id && e.targetPort === port.id);
  const inputEdge = override ?? base[isCrossAttention(source) && port.role !== 'Q' ? 1 : 0];
  const inputShape = inputEdge ? edgeOutputShape(graph, inputEdge, shapes) : shape;
  return inputShape ? [inputShape[0], inputShape[1], Number(source.params.embed_dim ?? 64) / cfg.heads] : undefined;
}
