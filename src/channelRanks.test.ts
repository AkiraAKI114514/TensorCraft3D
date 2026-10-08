import { describe, expect, it } from 'vitest';
import { analyze, validateGraph } from './analysis';

type Shape = number[];

function graph(op: string, shape: Shape, params: Record<string, unknown> = {}) {
  return {
    version: 1,
    name: `${op} rank contract`,
    nodes: [
      { id: 'input', name: 'Input', op: 'Input', params: { shape }, position: { x: 0, y: 0 } },
      { id: 'layer', name: op, op, params, position: { x: 100, y: 0 } },
      { id: 'output', name: 'Output', op: 'Output', params: {}, position: { x: 200, y: 0 } },
    ],
    edges: [
      { id: 'in', source: 'input', target: 'layer' },
      { id: 'out', source: 'layer', target: 'output' },
    ],
  };
}

describe('channel operation rank contracts', () => {
  it.each([
    ['BatchNorm1d', [2, 4], [2, 4]],
    ['BatchNorm1d', [2, 4, 6], [2, 4, 6]],
    ['BatchNorm2d', [2, 4, 3, 3], [2, 4, 3, 3]],
    ['BatchNorm3d', [2, 4, 2, 3, 3], [2, 4, 2, 3, 3]],
    ['ConvTranspose1d', [2, 3, 8], [2, 16, 16]],
    ['MaxPool1d', [2, 4, 8], [2, 4, 4]],
    ['AdaptiveAvgPool1d', [2, 4, 8], [2, 4, 2]],
  ] as [string, Shape, Shape][])('%s accepts rank-specific shape %j', (op, shape, expected) => {
    const validated = validateGraph(graph(op, shape, op === 'AdaptiveAvgPool1d' ? { output_size: 2 } : {}));
    const result = analyze(validated);
    expect(result.valid).toBe(true);
    expect(result.layers.output.output).toEqual(expected);
  });

  it('keeps InstanceNorm1d strict to rank three', () => {
    const result = analyze(validateGraph(graph('InstanceNorm1d', [2, 4])));
    expect(result.valid).toBe(false);
    expect(result.diagnostics.some(diagnostic => diagnostic.nodeId === 'layer' && diagnostic.code === 'SHAPE')).toBe(true);
  });
});
