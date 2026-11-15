import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { analyze, analyzeLevel, formatNumber, moeParameters, STRUCTURE_LIMITS, validateGraph } from './analysis';
import { graphAt } from './subgraph';
import { legendRows } from './operatorStyle';
import { manualGroup } from './grouping';
import { PRESETS } from './presets';

// The importer's graph for DeepSeek-V3's own configuration (backend/test_deepseek_reference.py pins
// the importer to this file): all 61 layers flat in one graph, 384 nodes.
const full = () => validateGraph(JSON.parse(readFileSync(new URL('../tests/fixtures/deepseek-v3-671b-structure.json', import.meta.url), 'utf8')));

describe('structure-only models', () => {
  it('analyses the full DeepSeek-V3 for its structure: 671B parameters, every shape, no run budget', () => {
    const graph = full(), analysis = analyze(graph);
    expect(graph.structureOnly).toBe(true);
    expect(analysis.valid).toBe(true);
    expect(analysis.parameters).toBe(671_026_404_352);
    expect(formatNumber(analysis.parameters)).toBe('671.03B');
    const output = analysis.order.at(-1)!;
    expect(analysis.layers[output].output).toEqual([1, 129280]);
    // One note says why nothing runs; nothing is an error.
    expect(analysis.diagnostics.map(d => [d.level, d.code])).toEqual([['info', 'VIEW_ONLY']]);
    // Every layer in one view: no folded block, all 61 attention layers and 58 MoE layers at the model width.
    expect(graph.nodes).toHaveLength(384);
    expect(graph.subgraphs).toBeUndefined();
    const layers = (op: string) => graph.nodes.filter(node => node.op === op);
    expect([layers('LatentAttention').length, layers('MoE').length, layers('Group').length]).toEqual([61, 58, 0]);
    expect(new Set([...layers('LatentAttention'), ...layers('MoE')].map(node => analysis.layers[node.id].output.join('×')))).toEqual(new Set(['1×16×7168']));
  });

  it('holds the same graph to the local limits once it is no longer marked', () => {
    const { structureOnly: _, ...runnable } = full(), analysis = analyze(runnable);
    expect(analysis.valid).toBe(false);
    expect(analysis.diagnostics.some(d => d.level === 'error')).toBe(true);
    // Nor may a project that runs hold 384 layers.
    expect(() => validateGraph(runnable)).toThrow('128 层');
  });

  it('holds a structure-only graph to its own, larger bounds', () => {
    const graph = full(), filler = graph.nodes.find(node => node.op === 'RMSNorm')!;
    const extra = Array.from({ length: STRUCTURE_LIMITS.nodes - graph.nodes.length + 1 }, (_, index) => ({ ...filler, id: `extra_${index}` }));
    expect(() => validateGraph({ ...graph, nodes: [...graph.nodes, ...extra] })).toThrow('512 层');
    expect(() => validateGraph({ ...graph, nodes: [...graph.nodes, ...extra.slice(1)] })).not.toThrow();
  });

  it('keeps 128 heads and 256 experts in every layer of the full model', () => {
    const graph = full(), analysis = analyze(graph);
    // 256 routed and 1 shared SwiGLU expert of 7168 → 2048 → 7168, and the 256 × 7168 gate: about 11.3B.
    const experts = graph.nodes.filter(node => node.op === 'MoE');
    expect(new Set(experts.map(node => analysis.layers[node.id].parameters))).toEqual(new Set([moeParameters(7168, 256, 1, 2048)]));
    expect(moeParameters(7168, 256, 1, 2048)).toBe(257 * 3 * 7168 * 2048 + 256 * 7168);
    expect(new Set(graph.nodes.filter(node => node.op === 'LatentAttention').map(node => node.params.num_heads))).toEqual(new Set([128]));
    expect(new Set(experts.map(node => `${node.params.experts}/${node.params.groups}`))).toEqual(new Set(['256/8']));
    expect(legendRows(graph, analysis).map(row => row.key)).toEqual(expect.arrayContaining(['latent', 'moe', 'residual']));
  });

  it('keeps a block grouped out of a structure-only model structure-only', () => {
    // A block needs one entry; a layer's attention half, norm and MLA, has one.
    const graph = full(), half = graph.nodes.filter(node => ['layers.3.attn_norm', 'layers.3.attn'].includes(node.name)).map(node => node.id);
    expect(half).toHaveLength(2);
    const grouped = manualGroup(graph, half);
    if ('error' in grouped) throw new Error(grouped.error);
    const instance = grouped.graph.nodes.find(node => node.id === grouped.id)!, block = graphAt(grouped.graph, [instance.subgraph!]);
    expect(block.structureOnly).toBe(true);
    // Drilled into, the block is analysed with what enters it: every layer has its shape, nothing is an error.
    const level = analyzeLevel(grouped.graph, [instance.subgraph!]);
    expect(level.valid).toBe(true);
    expect(level.diagnostics.filter(d => d.level === 'error')).toEqual([]);
    expect(level.diagnostics.map(d => d.code)).toContain('VIEW_ONLY');
  });

  it('analyses a drilled-in block of an ordinary model with what enters it, too', () => {
    const grouped = manualGroup(PRESETS.cnn(), ['layer_1', 'layer_2', 'layer_3']);
    if ('error' in grouped) throw new Error(grouped.error);
    const graph = grouped.graph, instance = graph.nodes.find(node => node.id === grouped.id)!, alone = analyze(graphAt(graph, [instance.subgraph!]));
    const level = analyzeLevel(graph, [instance.subgraph!]);
    expect(alone.valid).toBe(false);
    expect(level.valid).toBe(true);
    expect(Object.values(level.layers).every(layer => layer.output.length > 0)).toBe(true);
  });

  it('accepts only true for the mark', () => {
    expect(() => validateGraph({ ...full(), structureOnly: false })).toThrow('structureOnly');
  });
});
