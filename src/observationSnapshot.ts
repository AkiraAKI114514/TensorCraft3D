import { validateGraph } from './analysis';
import { ATTENTION_TENSORS, type AttentionMatrix, type AttentionSnapshot, type ChannelPage, type TensorSnapshot } from './inferenceTypes';
import { expandGraph } from './subgraph';
import { computationKey } from './trainedModel';
import type { Graph } from './types';

/**
 * An observation measured by a Python process outside the workbench
 * (`backend/observation_snapshot.py`): the graph it ran, plus the same bounded tensors the
 * inspector shows, for every node. The workbench only reads it. Nothing in it is executed, no
 * weights are loaded, and its numbers are shown only while the current graph computes exactly
 * what the snapshot's graph did.
 */
export interface ObservationSnapshot {
  format: typeof OBSERVATION_FORMAT;
  version: 1;
  snapshotId: string;
  createdAt: string;
  label: string | null;
  graph: Graph;
  device: 'CPU' | 'CUDA';
  seed: number;
  sampleIndex: 0;
  inputSource: 'synthetic' | 'provided';
  weights: 'external' | 'random-initialized';
  mode: 'eval';
  tensors: TensorSnapshot[];
  attentions: AttentionSnapshot[];
}

export const OBSERVATION_FORMAT = 'tensorcraft-observation';
/** Same bound as the writer's `MAX_SNAPSHOT_BYTES`; checked before the file is read. */
export const MAX_OBSERVATION_BYTES = 16 * 1024 * 1024;
const MAX_SIDE = 16, MAX_ENTRIES = 4096, MAX_BINS = 64;

/** Messages are i18n keys; `{field}` names the part of the file that was refused. */
export class ObservationSnapshotError extends Error {}

type Value = Record<string, unknown>;
const invalid = (field: string): never => { throw new ObservationSnapshotError(`观测快照内容无效：${field}`); };
const record = (value: unknown, field: string): Value => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Value : invalid(field);
const list = (value: unknown, field: string, max = MAX_ENTRIES): unknown[] => Array.isArray(value) && value.length <= max ? value : invalid(field);
const integer = (value: unknown, field: string, min = 0, max = 2 ** 31 - 1): number => typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max ? value : invalid(field);
const finite = (value: unknown, field: string): number => typeof value === 'number' && Number.isFinite(value) ? value : invalid(field);
const measured = (value: unknown, field: string): number | null => value === null ? null : finite(value, field);
const text = (value: unknown, field: string, max: number): string => typeof value === 'string' && value.length <= max ? value : invalid(field);
const oneOf = <T extends string | number>(value: unknown, options: readonly T[], field: string): T => options.includes(value as T) ? value as T : invalid(field);
const flag = (value: unknown, field: string): boolean => typeof value === 'boolean' ? value : invalid(field);
const shape = (value: unknown, field: string, maxRank: number) => list(value, field, maxRank).map((size, axis) => integer(size, `${field}[${axis}]`));
const matrix = (value: unknown, rows: number, columns: number, field: string) => {
  const parsed = list(value, field, MAX_SIDE).map((row, i) => list(row, `${field}[${i}]`, MAX_SIDE).map((item, j) => measured(item, `${field}[${i}][${j}]`)));
  return parsed.length === rows && parsed.every(row => row.length === columns) ? parsed : invalid(field);
};

function channelPage(value: unknown, field: string): ChannelPage {
  const page = record(value, field);
  const count = integer(page.count, `${field}.count`, 1, MAX_SIDE), total = integer(page.total, `${field}.total`, 1);
  const start = integer(page.start, `${field}.start`, 0, total - 1), selected = integer(page.selected, `${field}.selected`, start, start + count - 1);
  const rows = integer(page.rows, `${field}.rows`, 1, MAX_SIDE), columns = integer(page.columns, `${field}.columns`, 1, MAX_SIDE);
  if (start + count > total) invalid(`${field}.count`);
  const maps = list(page.maps, `${field}.maps`, MAX_SIDE).map((map, index) => matrix(map, rows, columns, `${field}.maps[${index}]`));
  const stats = list(page.stats, `${field}.stats`, MAX_SIDE).map((item, index) => { const stat = record(item, `${field}.stats[${index}]`); return { min: measured(stat.min, `${field}.stats[${index}].min`), max: measured(stat.max, `${field}.stats[${index}].max`), mean: measured(stat.mean, `${field}.stats[${index}].mean`) }; });
  if (maps.length !== count || stats.length !== count) invalid(`${field}.maps`);
  return { start, count, total, selected, shape: shape(page.shape, `${field}.shape`, 2), rows, columns, truncated: flag(page.truncated, `${field}.truncated`), maps, stats };
}

function tensor(value: unknown, field: string): TensorSnapshot {
  const item = record(value, field);
  const dims = shape(item.shape, `${field}.shape`, 5);
  const elements = integer(item.elements, `${field}.elements`, 0, Number.MAX_SAFE_INTEGER);
  const finiteCount = integer(item.finiteCount, `${field}.finiteCount`, 0, Number.MAX_SAFE_INTEGER), nonFiniteCount = integer(item.nonFiniteCount, `${field}.nonFiniteCount`, 0, Number.MAX_SAFE_INTEGER);
  if (elements !== dims.reduce((a, b) => a * b, 1) || finiteCount + nonFiniteCount !== elements) invalid(`${field}.elements`);
  const stats = record(item.stats, `${field}.stats`), histogram = record(item.histogram, `${field}.histogram`), slice = record(item.slice, `${field}.slice`);
  const counts = list(histogram.counts, `${field}.histogram.counts`, MAX_BINS).map((count, i) => integer(count, `${field}.histogram.counts[${i}]`, 0, Number.MAX_SAFE_INTEGER));
  const edges = list(histogram.edges, `${field}.histogram.edges`, MAX_BINS + 1).map((edge, i) => finite(edge, `${field}.histogram.edges[${i}]`));
  if (!counts.length || edges.length !== counts.length + 1) invalid(`${field}.histogram`);
  const rows = integer(slice.rows, `${field}.slice.rows`, 1, MAX_SIDE), columns = integer(slice.columns, `${field}.slice.columns`, 1, MAX_SIDE);
  const indices = list(slice.indices, `${field}.slice.indices`, 3).map((index, axis) => integer(index, `${field}.slice.indices[${axis}]`, 0, Math.max(0, (dims[axis] ?? 1) - 1)));
  if (indices.length !== Math.max(0, dims.length - 2)) invalid(`${field}.slice.indices`);
  const channels = item.channels === undefined || item.channels === null ? null : channelPage(item.channels, `${field}.channels`);
  // A page shares the main window's size and leading indices, so moving between its channels
  // is only a change of which map is shown.
  if (channels && (dims.length < 4 || channels.rows !== rows || channels.columns !== columns || channels.total !== dims[1] || channels.selected !== indices[1])) invalid(`${field}.channels`);
  return {
    nodeId: text(item.nodeId, `${field}.nodeId`, 512), shape: dims, dtype: text(item.dtype, `${field}.dtype`, 32), elements, finiteCount, nonFiniteCount,
    stats: { min: measured(stats.min, `${field}.stats.min`), max: measured(stats.max, `${field}.stats.max`), mean: measured(stats.mean, `${field}.stats.mean`), std: measured(stats.std, `${field}.stats.std`) },
    histogram: { edges, counts },
    slice: { indices, shape: shape(slice.shape, `${field}.slice.shape`, 2), rows, columns, values: matrix(slice.values, rows, columns, `${field}.slice.values`), truncated: flag(slice.truncated, `${field}.slice.truncated`) },
    channels,
  };
}

function attention(value: unknown, field: string): AttentionSnapshot {
  const item = record(value, field), tensors = record(item.tensors, `${field}.tensors`);
  const keys = Object.keys(ATTENTION_TENSORS) as (keyof typeof ATTENTION_TENSORS)[];
  if (Object.keys(tensors).length !== keys.length) invalid(`${field}.tensors`);
  const matrices = Object.fromEntries(keys.map(key => {
    const raw = record(tensors[key], `${field}.tensors.${key}`);
    const described: AttentionMatrix = { ...tensor(raw, `${field}.tensors.${key}`), rowStart: integer(raw.rowStart, `${field}.tensors.${key}.rowStart`), columnStart: integer(raw.columnStart, `${field}.tensors.${key}.columnStart`) };
    return [key, described];
  })) as Record<keyof typeof ATTENTION_TENSORS, AttentionMatrix>;
  const numHeads = integer(item.numHeads, `${field}.numHeads`, 1, 16), branches = integer(item.branches, `${field}.branches`, 1, 8), kvHeads = integer(item.kvHeads, `${field}.kvHeads`, 1, numHeads);
  return {
    nodeId: text(item.nodeId, `${field}.nodeId`, 512), branch: integer(item.branch, `${field}.branch`, 0, branches - 1), head: integer(item.head, `${field}.head`, 0, numHeads - 1),
    queryStart: integer(item.queryStart, `${field}.queryStart`), keyStart: integer(item.keyStart, `${field}.keyStart`), kvHead: integer(item.kvHead, `${field}.kvHead`, 0, kvHeads - 1),
    numHeads, kvHeads, branches, headDim: integer(item.headDim, `${field}.headDim`, 1), queryLength: integer(item.queryLength, `${field}.queryLength`, 1), keyLength: integer(item.keyLength, `${field}.keyLength`, 1),
    scale: finite(item.scale, `${field}.scale`), mask: oneOf(item.mask, ['none', 'causal'] as const, `${field}.mask`), dropout: oneOf(item.dropout, [0] as const, `${field}.dropout`),
    scoreSource: oneOf(item.scoreSource, ['projected-qk'] as const, `${field}.scoreSource`), outputSource: oneOf(item.outputSource, ['scaled_dot_product_attention'] as const, `${field}.outputSource`),
    tensors: matrices,
  };
}

/** Parse and check a snapshot file's text. Throws {@link ObservationSnapshotError} with an i18n key. */
export function parseObservationSnapshot(source: string): ObservationSnapshot {
  let value: unknown;
  try { value = JSON.parse(source); } catch { throw new ObservationSnapshotError('观测快照不是有效的 JSON'); }
  const file = value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Value : undefined;
  if (file?.format !== OBSERVATION_FORMAT) throw new ObservationSnapshotError('这不是 TensorCraft3D 观测快照');
  if (file.version !== 1) throw new ObservationSnapshotError('观测快照版本不受支持');
  let graph: Graph;
  try { graph = validateGraph(file.graph); } catch { return invalid('graph'); }
  // Observations are keyed by the expanded graph the runtime executed, so folded blocks resolve
  // to their inner layers exactly as the writer saw them.
  let nodes: Map<string, string>;
  try { nodes = new Map(expandGraph(graph).nodes.map(node => [node.id, node.op])); } catch { return invalid('graph'); }
  const tensors = list(file.tensors, 'tensors').map((item, index) => tensor(item, `tensors[${index}]`));
  const attentions = list(file.attentions, 'attentions').map((item, index) => attention(item, `attentions[${index}]`));
  const seen = new Set<string>();
  tensors.forEach((item, index) => { if (!nodes.has(item.nodeId) || seen.has(item.nodeId)) invalid(`tensors[${index}].nodeId`); seen.add(item.nodeId); });
  const attended = new Set<string>();
  attentions.forEach((item, index) => { const op = nodes.get(item.nodeId); if ((op !== 'MultiHeadAttention' && op !== 'Transformer') || attended.has(item.nodeId)) invalid(`attentions[${index}].nodeId`); attended.add(item.nodeId); });
  return {
    format: OBSERVATION_FORMAT, version: 1, snapshotId: text(file.snapshotId, 'snapshotId', 128), createdAt: text(file.createdAt, 'createdAt', 64),
    label: file.label === null || file.label === undefined ? null : text(file.label, 'label', 120), graph,
    device: oneOf(file.device, ['CPU', 'CUDA'] as const, 'device'), seed: integer(file.seed, 'seed'), sampleIndex: oneOf(file.sampleIndex, [0] as const, 'sampleIndex'),
    inputSource: oneOf(file.inputSource, ['synthetic', 'provided'] as const, 'inputSource'), weights: oneOf(file.weights, ['external', 'random-initialized'] as const, 'weights'),
    mode: oneOf(file.mode, ['eval'] as const, 'mode'), tensors, attentions,
  };
}

/** Whether the snapshot was measured on what `graph` computes now; names and layout do not count. */
export function snapshotMatches(graph: Graph, snapshot: ObservationSnapshot) {
  return computationKey(graph) === computationKey(snapshot.graph);
}

/** The same observation with its main window moved to another channel of the captured page. */
export function withChannel(snapshot: TensorSnapshot, channel: number): TensorSnapshot {
  const page = snapshot.channels;
  if (!page || channel < page.start || channel >= page.start + page.count || channel === page.selected) return snapshot;
  const indices = [...snapshot.slice.indices]; indices[1] = channel;
  return { ...snapshot, slice: { ...snapshot.slice, indices, values: page.maps[channel - page.start] }, channels: { ...page, selected: channel } };
}
