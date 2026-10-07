import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { analyze, validateGraph } from './analysis';

type ContractCase = {
  id: string;
  graph: unknown;
  expected: { valid: false } | { valid: true; outputShape: number[]; parameters: number };
};

const { cases } = JSON.parse(readFileSync(new URL('../tests/fixtures/analysis-contracts.json', import.meta.url), 'utf8')) as { cases: ContractCase[] };

describe('shared frontend/backend analysis contracts', () => {
  it.each(cases)('$id', ({ graph: value, expected }) => {
    const graph = validateGraph(value), analysis = analyze(graph);
    expect(analysis.valid).toBe(expected.valid);
    if (expected.valid) {
      const output = graph.nodes.find(node => node.op === 'Output')!;
      expect(analysis.layers[output.id].output).toEqual(expected.outputShape);
      expect(analysis.parameters).toBe(expected.parameters);
    }
  });
});
