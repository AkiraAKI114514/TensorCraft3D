import { EXPAND_SEP } from './subgraph';
import { MAX_SUBGRAPH_DEPTH, isGroup, repeatOf, type Graph, type Layer, type Metric, type WeightDistribution } from './types';

/** 一次选择最多对应这么多个展开层；超出的部分不再统计，界面不会为它卡住。 */
export const MAX_LAYER_METRIC_KEYS = 4096;
/** 曲线最多画这么多个点，更长的训练按等距抽样显示，数据表同样使用抽样后的点。 */
export const MAX_LAYER_METRIC_POINTS = 120;

export type LayerMetricField = 'layerGradients' | 'deadRelu';
export interface LayerPoint { epoch: number; mean: number; min: number; max: number; count: number; }

/**
 * 训练指标按「展开后的算子 id」记账（如 `block/0/n1`），而界面选中的是当前层级的节点。
 * 这里返回选中节点在展开图里对应的全部算子 id：
 *
 * - 顶层普通节点就是它自己，`repeat` 折叠的节点在后端也是一个键；
 * - 结构块是它每个实例里的全部算子；
 * - 下钻后选中的内部节点，是所有引用该子图的实例里的同名算子。
 */
export function layerMetricKeys(graph: Graph, drillPath: string[], nodeId: string): string[] {
  const definitions = graph.subgraphs ?? {}, keys: string[] = [];
  const leaves = (node: Layer, id: string, depth: number) => {
    if (keys.length >= MAX_LAYER_METRIC_KEYS || depth > MAX_SUBGRAPH_DEPTH) return;
    if (!isGroup(node)) { keys.push(id); return; }
    const definition = node.subgraph ? definitions[node.subgraph] : undefined;
    if (!definition) return;
    for (let index = 0; index < repeatOf(node); index++) for (const inner of definition.nodes) leaves(inner, `${id}${EXPAND_SEP}${index}${EXPAND_SEP}${inner.id}`, depth + 1);
  };
  const walk = (nodes: Layer[], prefix: string, depth: number) => {
    for (const node of nodes) {
      if (keys.length >= MAX_LAYER_METRIC_KEYS) return;
      const id = prefix ? `${prefix}${EXPAND_SEP}${node.id}` : node.id;
      if (depth === drillPath.length) { if (node.id === nodeId) leaves(node, id, depth); continue; }
      if (!isGroup(node) || node.subgraph !== drillPath[depth]) continue;
      const definition = definitions[node.subgraph];
      if (!definition) continue;
      for (let index = 0; index < repeatOf(node); index++) walk(definition.nodes, `${id}${EXPAND_SEP}${index}`, depth + 1);
    }
  };
  if (drillPath.length <= MAX_SUBGRAPH_DEPTH) walk(graph.nodes, '', 0);
  return keys;
}

/**
 * 把若干展开层在同一轮的取值合成一个点。某一轮没有任何有限取值时不产生点，
 * 所以演示数据、只含五列的导入指标、以及没有该层记录的训练都得到空序列。
 */
export function layerSeries(metrics: Metric[], keys: string[], field: LayerMetricField): LayerPoint[] {
  const points: LayerPoint[] = [];
  for (const metric of metrics) {
    const record: unknown = metric[field];
    if (!record || typeof record !== 'object' || Array.isArray(record)) continue;
    let sum = 0, min = Infinity, max = -Infinity, count = 0;
    for (const key of keys) {
      const value = Object.hasOwn(record, key) ? (record as Record<string, unknown>)[key] : undefined;
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      sum += value; min = Math.min(min, value); max = Math.max(max, value); count += 1;
    }
    if (count) points.push({ epoch: metric.epoch, mean: sum / count, min, max, count });
  }
  return points;
}

export interface DistributionPoint extends WeightDistribution { epoch: number; }

/** 训练每轮推送的分位数个数：最小值、正态 ±1.5σ/±1σ/±0.5σ 对应分位、中位数与最大值。 */
export const DISTRIBUTION_QUANTILES = 9;
const MAX_DISTRIBUTION_BINS = 64;

function distributionAt(metric: Metric, key: string): WeightDistribution | null {
  const record: unknown = metric.weightDistributions;
  if (!record || typeof record !== 'object' || Array.isArray(record) || !Object.hasOwn(record, key)) return null;
  const entry: unknown = (record as Record<string, unknown>)[key];
  if (!entry || typeof entry !== 'object') return null;
  const { count, mean, std, quantiles, counts } = entry as Record<string, unknown>;
  const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
  if (!finite(count) || !finite(mean) || !finite(std)) return null;
  if (!Array.isArray(quantiles) || quantiles.length !== DISTRIBUTION_QUANTILES || !quantiles.every((value, index) => finite(value) && (index === 0 || value >= quantiles[index - 1]))) return null;
  if (!Array.isArray(counts) || !counts.length || counts.length > MAX_DISTRIBUTION_BINS || !counts.every(value => finite(value) && value >= 0)) return null;
  return { count, mean, std, quantiles, counts };
}

/**
 * 一个展开层每轮结束时的权重分布。分位数不能跨层合并，所以这里只取一个层键；
 * 导入或损坏的记录（分位数不是 9 个递增的有限值、计数不是有限非负数）整轮跳过。
 */
export function distributionSeries(metrics: Metric[], key: string): DistributionPoint[] {
  const points: DistributionPoint[] = [];
  for (const metric of metrics) {
    const entry = distributionAt(metric, key);
    if (entry) points.push({ epoch: metric.epoch, ...entry });
  }
  return points;
}

/** `keys` 中至少有一轮记录了权重分布的展开层，保持原顺序；记录是否有效由 `distributionSeries` 判断。 */
export function distributionKeys(metrics: Metric[], keys: string[]): string[] {
  const present = new Set<string>();
  for (const metric of metrics) {
    const record: unknown = metric.weightDistributions;
    if (!record || typeof record !== 'object' || Array.isArray(record)) continue;
    for (const key of keys) if (Object.hasOwn(record, key)) present.add(key);
  }
  return keys.filter(key => present.has(key));
}

/** 等距抽样到至多 `limit` 个点，始终保留首尾，所以最新一轮永远可读。 */
export function samplePoints<T>(points: T[], limit = MAX_LAYER_METRIC_POINTS): T[] {
  if (points.length <= limit) return points;
  const step = (points.length - 1) / (limit - 1);
  return Array.from({ length: limit }, (_, index) => points[Math.round(index * step)]);
}

/** 梯度范数跨越多个数量级，固定小数位会把小值显示成 0。 */
export function formatMetricValue(value: number, percent = false): string {
  if (percent) return `${(value * 100).toFixed(1)}%`;
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  return magnitude < 1e-3 || magnitude >= 1e4 ? value.toExponential(2) : value.toPrecision(3);
}
