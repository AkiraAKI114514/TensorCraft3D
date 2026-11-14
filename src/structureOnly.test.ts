import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { analyze, analyzeLevel, formatNumber, moeParameters, validateGraph } from './analysis';
import { graphAt } from './subgraph';
import { legendRows } from './operatorStyle';
import { manualGroup } from './grouping';
import { PRESETS } from './presets';

// The importer's graph for DeepSeek-V3's own configuration (backend/test_deepseek_reference.py pins
// the importer to this file): 61 layers folded into a dense block ×3 and an MoE block ×58.
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
    const blocks = graph.nodes.filter(node => node.op === 'Group').map(node => [node.repeat, analysis.layers[node.id].output]);
    expect(blocks).toEqual([[3, [1, 16, 7168]], [58, [1, 16, 7168]]]);
  });

  it('holds the same graph to the local limits once it is no longer marked', () => {
    const { structureOnly: _, ...runnable } = full(), analysis = analyze(runnable);
    expect(analysis.valid).toBe(false);
    expect(analysis.diagnostics.some(d => d.level === 'error')).toBe(true);
  });

  it('keeps a block of a structure-only model structure-only, with 128 heads and 256 experts', () => {
    const graph = full(), block = graphAt(graph, ['moe_layers']), analysis = analyze(block);
    expect(block.structureOnly).toBe(true);
    // A level is analysed without the tensor that enters it, so its own note is what the flag adds.
    expect(analysis.diagnostics.filter(d => d.code === 'VIEW_ONLY')).toHaveLength(1);
    // Drilled into, the block is analysed with what enters it: every layer has its shape, nothing is an error.
    const level = analyzeLevel(graph, ['moe_layers']);
    expect(level.valid).toBe(true);
    expect(level.diagnostics.filter(d => d.level === 'error')).toEqual([]);
    expect(level.diagnostics.map(d => d.code)).toContain('VIEW_ONLY');
    expect(block.nodes.map(node => level.layers[node.id].output)).toEqual(block.nodes.map(() => [1, 16, 7168]));
    // 256 routed and 1 shared SwiGLU expert of 7168 → 2048 → 7168, and the 256 × 7168 gate: about 11.3B.
    expect(level.layers[block.nodes.find(node => node.op === 'MoE')!.id].parameters).toBe(moeParameters(7168, 256, 1, 2048));
    expect(moeParameters(7168, 256, 1, 2048)).toBe(257 * 3 * 7168 * 2048 + 256 * 7168);
    const params = (op: string) => block.nodes.find(node => node.op === op)!.params;
    expect([params('LatentAttention').num_heads, params('MoE').experts, params('MoE').groups]).toEqual([128, 256, 8]);
    expect(legendRows(block, analysis).map(row => row.key)).toEqual(expect.arrayContaining(['latent', 'moe', 'residual']));
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

  it('ships as the DeepSeek-V3 671B template: the graph the importer builds', () => {
    const template = PRESETS.deepseek_v3();
    expect(template).toEqual({ ...full(), name: 'DeepSeek-V3 671B' });
    expect(analyze(template).parameters).toBe(671_026_404_352);
    // Each load is a fresh copy, so editing one never changes the next.
    expect(PRESETS.deepseek_v3().nodes).not.toBe(template.nodes);
  });

  it('accepts only true for the mark', () => {
    expect(() => validateGraph({ ...full(), structureOnly: false })).toThrow('structureOnly');
  });
});
