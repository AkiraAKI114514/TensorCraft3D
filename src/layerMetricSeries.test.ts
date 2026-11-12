import { describe, expect, it } from 'vitest';
import { formatMetricValue, layerMetricKeys, layerSeries, samplePoints, MAX_LAYER_METRIC_KEYS } from './layerMetricSeries';
import { expandGraph } from './subgraph';
import type { Graph, Layer, Metric } from './types';

const layer = (id: string, op: Layer['op'], extra: Partial<Layer> = {}): Layer => ({ id, name: id, op, params: {}, position: { x: 0, y: 0 }, ...extra });

// 外层块重复两次，块里又嵌着一个重复三次的内层块：覆盖顶层、块、下钻一层和下钻两层四种选择。
function nested(): Graph {
  return {
    version: 1, name: 'Nested blocks',
    nodes: [layer('in', 'Input', { params: { shape: [2, 4] } }), layer('outer', 'Group', { subgraph: 'block', repeat: 2 }), layer('head', 'Linear', { params: { out_features: 2 } }), layer('out', 'Output')],
    edges: [{ id: 'e0', source: 'in', target: 'outer' }, { id: 'e1', source: 'outer', target: 'head' }, { id: 'e2', source: 'head', target: 'out' }],
    subgraphs: {
      block: { id: 'block', name: 'Block', origin: 'manual', nodes: [layer('n0', 'Linear', { params: { out_features: 4 } }), layer('n1', 'Group', { subgraph: 'unit', repeat: 3 })], edges: [{ id: 'b0', source: 'n0', target: 'n1' }] },
      unit: { id: 'unit', name: 'Unit', origin: 'manual', nodes: [layer('n0', 'Linear', { params: { out_features: 4 } }), layer('n1', 'ReLU')], edges: [{ id: 'u0', source: 'n0', target: 'n1' }] },
    },
  };
}

const metric = (epoch: number, layerGradients?: Record<string, number>, deadRelu?: Record<string, number>): Metric => ({ epoch, trainLoss: 1, valLoss: 1, accuracy: 0.5, gradNorm: 1, layerGradients, deadRelu, source: 'training' });

describe('layer metric keys', () => {
  it('maps a plain top-level layer to itself', () => {
    expect(layerMetricKeys(nested(), [], 'head')).toEqual(['head']);
    expect(layerMetricKeys(nested(), [], 'missing')).toEqual([]);
  });

  it('expands a block into every operator of every instance, exactly as the trained graph names them', () => {
    const graph = nested(), keys = layerMetricKeys(graph, [], 'outer');
    // 训练端的键来自展开图，所以这里直接和 expandGraph 的结果对齐，而不是手写期望的字符串。
    const expanded = expandGraph(graph).nodes.map(node => node.id).filter(id => id.startsWith('outer/'));
    expect(keys.sort()).toEqual(expanded.sort());
    expect(keys).toHaveLength(2 * (1 + 3 * 2));
  });

  it('follows the drill path to the same-named operator in every instance', () => {
    const graph = nested(), expanded = new Set(expandGraph(graph).nodes.map(node => node.id));
    expect(layerMetricKeys(graph, ['block'], 'n0')).toEqual(['outer/0/n0', 'outer/1/n0']);
    const inner = layerMetricKeys(graph, ['block', 'unit'], 'n0');
    expect(inner).toEqual(['outer/0/n1/0/n0', 'outer/0/n1/1/n0', 'outer/0/n1/2/n0', 'outer/1/n1/0/n0', 'outer/1/n1/1/n0', 'outer/1/n1/2/n0']);
    for (const key of inner) expect(expanded.has(key), key).toBe(true);
    // 两个子图里都有 n0：按路径匹配，下钻到 block 时不能把 unit 里的 n0 算进来。
    expect(layerMetricKeys(graph, ['block'], 'n0')).not.toContain('outer/0/n1/0/n0');
    // 路径上不存在的子图没有对应的层。
    expect(layerMetricKeys(graph, ['unit'], 'n0')).toEqual([]);
  });

  it('stays bounded for a heavily folded block', () => {
    const graph = nested();
    graph.nodes[1].repeat = 1024; graph.subgraphs!.block.nodes[1].repeat = 1024;
    expect(layerMetricKeys(graph, [], 'outer').length).toBeLessThanOrEqual(MAX_LAYER_METRIC_KEYS);
    expect(layerMetricKeys(graph, ['block', 'unit'], 'n1').length).toBeLessThanOrEqual(MAX_LAYER_METRIC_KEYS);
  });
});

describe('layer metric series', () => {
  it('combines the selected layers per epoch and reports their spread', () => {
    const metrics = [metric(1, { a: 2, b: 4, other: 100 }), metric(2, { a: 1, b: 1 })];
    expect(layerSeries(metrics, ['a', 'b'], 'layerGradients')).toEqual([
      { epoch: 1, mean: 3, min: 2, max: 4, count: 2 },
      { epoch: 2, mean: 1, min: 1, max: 1, count: 2 },
    ]);
    expect(layerSeries(metrics, ['a'], 'deadRelu')).toEqual([]);
  });

  it('produces no point for epochs that never measured the layer', () => {
    const metrics: Metric[] = [metric(1), metric(2, { a: 0.5 }), metric(3, { b: 9 }), { ...metric(4), source: 'demo' }];
    expect(layerSeries(metrics, ['a'], 'layerGradients')).toEqual([{ epoch: 2, mean: 0.5, min: 0.5, max: 0.5, count: 1 }]);
    expect(layerSeries(metrics, [], 'layerGradients')).toEqual([]);
  });

  it('ignores malformed imported records instead of plotting them', () => {
    // 导入的指标只校验了五个数值列，逐层字段可能是任何 JSON。
    const imported = [
      { ...metric(1), layerGradients: 'oops' }, { ...metric(2), layerGradients: { a: 'NaN', b: null } },
      { ...metric(3), layerGradients: { a: Infinity } }, { ...metric(4), layerGradients: [1, 2] }, { ...metric(5), layerGradients: { a: 0.25 } },
    ] as unknown as Metric[];
    expect(layerSeries(imported, ['a', 'b', 'constructor', '0', 'length'], 'layerGradients')).toEqual([{ epoch: 5, mean: 0.25, min: 0.25, max: 0.25, count: 1 }]);
  });

  it('samples long runs evenly and always keeps the first and the latest epoch', () => {
    const points = Array.from({ length: 1000 }, (_, index) => index + 1);
    const sampled = samplePoints(points, 120);
    expect(sampled).toHaveLength(120);
    expect(sampled[0]).toBe(1);
    expect(sampled.at(-1)).toBe(1000);
    expect(new Set(sampled).size).toBe(120);
    expect(samplePoints([1, 2, 3], 120)).toEqual([1, 2, 3]);
  });

  it('keeps small gradient norms readable instead of rounding them to zero', () => {
    expect(formatMetricValue(0)).toBe('0');
    expect(formatMetricValue(0.01234)).toBe('0.0123');
    expect(formatMetricValue(3.4e-6)).toBe('3.40e-6');
    expect(formatMetricValue(123456)).toBe('1.23e+5');
    expect(formatMetricValue(0.9123, true)).toBe('91.2%');
  });
});
