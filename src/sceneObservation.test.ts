import { describe, expect, it } from 'vitest';
import type { InferenceReport, TensorSnapshot } from './inferenceTypes';
import { MISSING, blockPaint, observationFromReport } from './sceneObservation';
import { computationKey } from './trainedModel';
import { PRESETS } from './presets';

const tensor = (shape: number[], values: (number | null)[][], extra: Partial<TensorSnapshot> = {}): TensorSnapshot => ({
  nodeId: 'n', shape, dtype: 'float32', elements: shape.reduce((a, b) => a * b, 1), finiteCount: 0, nonFiniteCount: 0,
  stats: { min: -2, max: 0.5, mean: 0, std: 1 }, histogram: { edges: [0, 1], counts: [0] },
  slice: { indices: [], shape: [], rows: values.length, columns: values[0].length, values, truncated: false }, ...extra,
});

describe('measured values on the 3D blocks', () => {
  it('lays a feature map window on the large face and channel means through the thickness', () => {
    const channels = { start: 0, count: 2, total: 16, selected: 0, shape: [32, 32], rows: 2, columns: 2, truncated: true, maps: [], stats: [{ min: 0, max: 1, mean: 0.25 }, { min: null, max: null, mean: null }] };
    const paint = blockPaint(tensor([1, 16, 32, 32], [[1, null], [3, 4]], { channels }))!;
    expect(paint.map).toMatchObject({ columns: 2, rows: 2, u: 2, v: 1, uTotal: 32, vTotal: 32 });
    // Row-major, so texel (column, row) is the inspector's cell; non-finite cells get their own marker.
    expect([...paint.map!.values]).toEqual([1, MISSING, 3, 4]);
    expect(paint.strip).toMatchObject({ axis: 0, total: 16, start: 0, count: 2 });
    expect([...paint.strip!.values]).toEqual([0.25, MISSING]);
    // The colour range is the whole tensor's, not the window's.
    expect(paint.limit).toBe(2);
  });

  it('puts a sequence\'s positions across and features down, and a vector\'s features down', () => {
    expect(blockPaint(tensor([1, 5, 8], [[0, 1]]))!.map).toMatchObject({ u: 1, v: 2, uTotal: 8, vTotal: 5 });
    expect(blockPaint(tensor([1, 10], [[0, 1, 2]]))).toMatchObject({ map: { u: 1, v: 2, uTotal: 10, vTotal: 1 }, strip: null });
    expect(blockPaint(tensor([1, 2, 3, 4, 5], [[0]]))).toBeNull();
  });

  it('binds a live run to the computation it measured', () => {
    const graph = PRESETS.mlp();
    const report = { runId: 'run', createdAt: 'now', weights: 'trained', tensors: [tensor([1, 4], [[1, 2]], { nodeId: 'a' })] } as unknown as InferenceReport;
    const observation = observationFromReport(report, graph);
    expect(observation).toMatchObject({ key: 'run', source: 'live', weights: 'trained', graphKey: computationKey(graph) });
    expect(Object.keys(observation.tensors)).toEqual(['a']);
  });
});
