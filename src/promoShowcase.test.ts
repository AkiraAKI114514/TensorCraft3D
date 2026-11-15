import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { analyze, validateGraph } from './analysis';
import type { Graph } from './types';

// The promo video (scripts/promo) flies through these graphs; a shape error would film red layers.
const DIR = new URL('../scripts/promo/graphs/', import.meta.url);
const graphs = readdirSync(DIR).filter(name => name.endsWith('.json'))
  .map(name => [name, JSON.parse(readFileSync(new URL(name, DIR), 'utf8')) as Graph] as const);

describe('promo showcase graphs', () => {
  it('are all there', () => expect(graphs.map(([name]) => name).sort()).toEqual(['showcase-attention.json', 'showcase-sequence.json', 'showcase-signal.json', 'showcase-vision.json']));
  it.each(graphs)('%s analyses without errors', (_, graph) => {
    const result = analyze(validateGraph(graph));
    expect(result.diagnostics.filter(d => d.level === 'error')).toEqual([]);
    expect(result.valid).toBe(true);
  });
});
