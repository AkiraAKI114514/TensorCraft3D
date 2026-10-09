import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { analyze, validateGraph } from './analysis';
import { repeatOf, type Graph } from './types';

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

  // Folding is only linear in parameters: for every valid fixture that folds a
  // node, removing `repeat` must scale that node's contribution back to one
  // instance while leaving the output shape untouched.
  const folded = cases.filter(({ graph, expected }) => expected.valid && (graph as Graph).nodes.some(node => repeatOf(node) > 1));
  it('folds every repeat-bearing fixture case without changing shapes', () => {
    expect(folded.length).toBeGreaterThan(0);
    for (const { id, graph: value } of folded) {
      const graph = validateGraph(value), foldedAnalysis = analyze(graph);
      const target = graph.nodes.find(node => repeatOf(node) > 1)!;
      const factor = repeatOf(target);
      const stripped = validateGraph({ ...graph, nodes: graph.nodes.map(({ repeat: _repeat, ...node }) => node) });
      const baseline = analyze(stripped);
      expect(foldedAnalysis.layers[target.id].output).toEqual(baseline.layers[target.id].output);
      expect(foldedAnalysis.layers[target.id].parameters).toBe(baseline.layers[target.id].parameters * factor);
      expect(foldedAnalysis.parameters).toBe(baseline.parameters + (factor - 1) * baseline.layers[target.id].parameters);
      expect(id).toBeTruthy();
    }
  });
});

