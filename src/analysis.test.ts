import { describe, it, expect } from 'vitest';
import { analyze, diagnoseMetrics, validateGraph } from './analysis';
import { PRESETS } from './presets';
import { generatePython } from './export';
import type { Metric } from './types';

describe('model contracts', () => {
  it.each(['cnn', 'mlp', 'residual'])('infers a valid %s model', preset => {
    const graph = PRESETS[preset](), analysis = analyze(graph);
    expect(analysis.valid).toBe(true);
    expect(analysis.layers[graph.nodes.at(-1)!.id].output).toEqual(preset === 'mlp' ? [1, 3] : [1, 10]);
    expect(analysis.parameters).toBeGreaterThan(0);
  });
  it('rejects linear projection of an unflattened image', () => {
    const graph = PRESETS.blank(); graph.nodes[1].op = 'Linear'; graph.nodes[1].params = { out_features: 3 };
    expect(analyze(graph).diagnostics.some(d => d.code === 'SHAPE' && d.nodeId === 'layer_1')).toBe(true);
    expect(() => generatePython(graph)).toThrow();
  });
  it('rejects a residual branch with incompatible channels', () => {
    const graph = PRESETS.residual(); graph.nodes[3].params.out_channels = 32;
    expect(analyze(graph).diagnostics.some(d => d.nodeId === 'layer_5' && d.code === 'SHAPE')).toBe(true);
  });
  it('detects a directed cycle and duplicate edges', () => {
    const graph = PRESETS.cnn(); graph.edges.push({ id: 'cycle', source: 'layer_7', target: 'layer_1' });
    expect(analyze(graph).diagnostics.some(d => d.code === 'CYCLE')).toBe(true);
    const duplicate = PRESETS.cnn(); duplicate.edges.push({ ...duplicate.edges[0], id: 'duplicate' });
    expect(() => validateGraph(duplicate)).toThrow();
  });
  it('exports branch operations with defined values', () => {
    const code = generatePython(PRESETS.residual());
    expect(code).toContain('values["layer_5"] = values["layer_4"] + values["layer_2"]');
    expect(code).toContain('nn.Linear(16, 10)');
    expect(code).not.toContain('undefined');
  });
  it('infers Transformer dimensions and exact parameter counts', () => {
    const graph = PRESETS.transformer(), analysis = analyze(graph);
    expect(analysis.valid).toBe(true);
    expect(analysis.layers.layer_1.output).toEqual([1, 16, 64]);
    expect(analysis.layers.layer_1.parameters).toBe(33472);
    expect(analysis.layers.layer_5.output).toEqual([1, 10]);
    expect(generatePython(graph)).toContain('norm_first=True');
  });
  it('rejects mismatched embeddings, indivisible heads and attention memory overflow', () => {
    const graph = PRESETS.transformer(); graph.nodes[1].params.num_heads = 3;
    expect(analyze(graph).valid).toBe(false);
    graph.nodes[1].params.num_heads = 4; graph.nodes[1].params.kv_heads = 4; graph.nodes[1].params.embed_dim = 32;
    expect(analyze(graph).valid).toBe(false);
    graph.nodes[1].params.embed_dim = 64; graph.nodes[0].params.shape = [1, 4096, 64];
    expect(analyze(graph).diagnostics.some(d => d.message.includes('注意力矩阵'))).toBe(true);
  });
  it('exports standalone multi-head self-attention without tuple outputs', () => {
    const graph = PRESETS.transformer(); graph.nodes[1].op = 'MultiHeadAttention';
    expect(analyze(graph).layers.layer_1.parameters).toBe(16640);
    expect(generatePython(graph)).toContain('F.scaled_dot_product_attention(q, k, v');
    expect(generatePython(graph)).toContain('TensorLabAttention(embed_dim=64');
  });
  it('ignores retired polygon counts in legacy projects', () => {
    const graph = PRESETS.transformer(), original = analyze(graph), code = generatePython(graph);
    graph.nodes[1].params.qkv_count = 6;
    expect(analyze(graph).layers.layer_1).toEqual(original.layers.layer_1);
    expect(generatePython(graph)).toBe(code);
    for (const count of [2, 65, 4.5]) {
      graph.nodes[1].params.qkv_count = count;
      expect(analyze(graph).valid).toBe(true);
    }
  });
  it('rejects illegal input/output connections and ordinary multi-input layers', () => {
    const graph = PRESETS.mlp();
    graph.edges.push({ id: 'extra', source: 'layer_0', target: 'layer_2' });
    expect(analyze(graph).diagnostics.some(d => d.nodeId === 'layer_2' && d.message.includes('一个输入'))).toBe(true);
    const output = PRESETS.mlp();
    output.nodes.push({ id: 'extra', name: 'extra', op: 'ReLU', params: {}, position: { x: 2000, y: 100 } });
    output.edges.push({ id: 'invalid', source: 'layer_7', target: 'extra' });
    expect(analyze(output).diagnostics.some(d => d.nodeId === 'layer_7' && d.message.includes('不能连接下游'))).toBe(true);
    const input = PRESETS.blank();
    input.nodes.push({ ...input.nodes[0], id: 'input2' });
    expect(analyze(input).diagnostics.some(d => d.code === 'INPUT_UNUSED')).toBe(true);
  });
});

describe('training alerts', () => {
  const metric = (i: number, trainLoss: number, valLoss: number): Metric => ({ epoch: i + 1, trainLoss, valLoss, accuracy: 0.5, gradNorm: 0.1, source: 'training' });
  it('flags widening train/validation gap', () => {
    const values = Array.from({ length: 8 }, (_, i) => metric(i, 0.7 - i * 0.05, 0.9 + i * 0.04));
    expect(diagnoseMetrics(values).some(d => d.code === 'OVERFIT')).toBe(true);
  });
  it('flags a plateau but does not flag steadily improving losses', () => {
    expect(diagnoseMetrics(Array.from({ length: 8 }, (_, i) => metric(i, 1, 1.1))).some(d => d.code === 'NOT_CONVERGING')).toBe(true);
    expect(diagnoseMetrics(Array.from({ length: 8 }, (_, i) => metric(i, 1 - i * 0.08, 1.1 - i * 0.08)))).toEqual([]);
  });
  it('flags NaN and large per-layer gradients', () => {
    const value = { ...metric(1, NaN, 1), layerGradients: { conv: 120 } };
    expect(diagnoseMetrics([value]).map(d => d.code)).toEqual(['NON_FINITE', 'LAYER_GRADIENT']);
  });
});
