import { describe, it, expect } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze, diagnoseMetrics, product, validateGraph } from './analysis';
import { PRESETS } from './presets';
import { generatePython } from './export';
import { DEFAULTS, MAX_REPEAT, repeatOf, type Graph, type Metric, type Params } from './types';

describe('model contracts', () => {
  it.each(['cnn', 'mlp', 'residual'])('infers a valid %s model', preset => {
    const graph = PRESETS[preset](), analysis = analyze(graph);
    expect(analysis.valid).toBe(true);
    expect(analysis.layers[graph.nodes.at(-1)!.id].output).toEqual(preset === 'mlp' ? [1, 3] : [1, 10]);
    expect(analysis.parameters).toBeGreaterThan(0);
  });
  it.each([[2, 101], [64, 6, 101], [64, 6, 69], [2, 3, 4, 5], [2, 3, 4, 5, 6]])('projects the final dimension of Linear input %j', (...shape) => {
    const graph = PRESETS.blank(); graph.nodes[0].params.shape = shape;
    graph.nodes.splice(1, 0, { id: 'projection', name: 'Projection', op: 'Linear', params: { out_features: 64 }, position: { x: 100, y: 100 } });
    graph.edges = [{ id: 'in', source: 'layer_0', target: 'projection' }, { id: 'out', source: 'projection', target: 'layer_1' }];
    const analysis = analyze(graph);
    expect(analysis.valid).toBe(true);
    expect(analysis.layers.layer_1.output).toEqual([...shape.slice(0, -1), 64]);
    expect(analysis.parameters).toBe((shape.at(-1)! + 1) * 64);
    expect(generatePython(graph)).toContain(`nn.Linear(${shape.at(-1)}, 64)`);
  });
  it('diagnoses Linear bottlenecks by feature width rather than sequence length', () => {
    const graph = PRESETS.mlp(); graph.nodes[0].params.shape = [2, 6, 256]; graph.nodes[1].params.out_features = 8;
    expect(analyze(graph).valid).toBe(true);
    expect(analyze(graph).diagnostics.some(d => d.code === 'BOTTLENECK' && d.nodeId === 'layer_1')).toBe(true);
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
  it.each([
    [DEFAULTS.Bilinear, 16],
    [{ out_features: 10 }, 16],
    [{ in2_features: 7, out_features: 10 }, 7],
  ] as [Params, number][])('builds Bilinear with parameters %j and a second input width of %i', (params, width) => {
    const graph = PRESETS.blank();
    graph.nodes[0].params.shape = [4, 8];
    graph.nodes.splice(1, 0, { id: 'bilinear', name: 'Bilinear', op: 'Bilinear', params: { ...params }, position: { x: 100, y: 100 } });
    graph.nodes.push({ id: 'second', name: 'Second', op: 'Input', params: { shape: [4, width] }, position: { x: 0, y: 320 } });
    graph.edges = [{ id: 'first', source: 'layer_0', target: 'bilinear' }, { id: 'second', source: 'second', target: 'bilinear' }, { id: 'out', source: 'bilinear', target: 'layer_1' }];
    const analysis = analyze(graph);
    expect(analysis.valid).toBe(true);
    expect(analysis.layers.bilinear.output).toEqual([4, 10]);
    expect(analysis.layers.bilinear.parameters).toBe(8 * width * 10 + 10);
    expect(generatePython(graph)).toContain(`nn.Bilinear(8, ${width}, 10)`);
    graph.nodes.find(n => n.id === 'bilinear')!.params.in2_features = 8;
    expect(analyze(graph).valid).toBe(false);
    expect(analyze(graph).diagnostics.some(d => d.nodeId === 'bilinear' && d.message.includes('in2_features'))).toBe(true);
    expect(() => generatePython(graph)).toThrow();
    delete graph.nodes.find(n => n.id === 'bilinear')!.params.in2_features;
    graph.nodes.find(n => n.id === 'second')!.params.shape = [4, 8];
    expect(analyze(graph).valid).toBe(false);
    expect(() => generatePython(graph)).toThrow();
    graph.nodes.find(n => n.id === 'second')!.params.shape = [5, 16];
    expect(analyze(graph).diagnostics.some(d => d.nodeId === 'bilinear' && d.message.includes('batch'))).toBe(true);
    graph.nodes.find(n => n.id === 'second')!.params.shape = [4, 2, 16];
    expect(analyze(graph).diagnostics.some(d => d.nodeId === 'bilinear' && d.message.includes('二维'))).toBe(true);
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

describe('repeat folding', () => {
  const foldedLinear = (repeat?: number): Graph => {
    const graph = PRESETS.blank();
    graph.nodes[0].params.shape = [2, 4];
    graph.nodes.splice(1, 0,
      { id: 'hidden', name: 'Hidden', op: 'Linear', params: { out_features: 4 }, position: { x: 100, y: 100 }, ...(repeat === undefined ? {} : { repeat }) },
      { id: 'act', name: 'Act', op: 'ReLU', params: {}, position: { x: 200, y: 100 } },
      { id: 'head', name: 'Head', op: 'Linear', params: { out_features: 2 }, position: { x: 300, y: 100 } });
    graph.edges = [
      { id: 'e0', source: 'layer_0', target: 'hidden' },
      { id: 'e1', source: 'hidden', target: 'act' },
      { id: 'e2', source: 'act', target: 'head' },
      { id: 'e3', source: 'head', target: 'layer_1' },
    ];
    return graph;
  };

  it('treats an absent repeat as one instance and keeps legacy output byte-identical', () => {
    expect(repeatOf({})).toBe(1);
    const baseline = analyze(foldedLinear());
    expect(baseline.valid).toBe(true);
    expect(baseline.parameters).toBe(30);
    expect(baseline.layers.hidden).toEqual({ input: [[2, 4]], output: [2, 4], parameters: 20, repeat: 1 });
    // repeat: 1 and repeat: undefined must be indistinguishable to the analyzer, the
    // parameter totals and the generated Python (no drift for existing projects).
    expect(analyze(foldedLinear(1))).toEqual(baseline);
    expect(generatePython(foldedLinear(1))).toBe(generatePython(foldedLinear()));
  });

  it('scales parameters and activation memory by N while preserving shapes', () => {
    const baseline = analyze(foldedLinear());
    for (const repeat of [2, 3, 17, MAX_REPEAT]) {
      const analysis = analyze(foldedLinear(repeat));
      expect(analysis.valid).toBe(true);
      expect(analysis.layers.hidden.parameters).toBe(20 * repeat);
      expect(analysis.layers.hidden.repeat).toBe(repeat);
      expect(analysis.layers.head.parameters).toBe(10);
      expect(analysis.parameters).toBe(20 * repeat + 10);
      // Instances are chained, so the folded node's output shape is unchanged.
      expect(analysis.layers.hidden.output).toEqual(baseline.layers.hidden.output);
      expect(analysis.layers.head.output).toEqual(baseline.layers.head.output);
      expect(analysis.activationBytes).toBe(baseline.activationBytes + product([2, 4]) * 4 * (repeat - 1));
    }
  });

  it('rejects repeats that are zero, negative, fractional or above MAX_REPEAT', () => {
    for (const repeat of [0, -1, MAX_REPEAT + 1, 2.5, 1e9]) {
      expect(() => validateGraph(foldedLinear(repeat))).toThrow(/repeat/);
    }
    expect(() => validateGraph(foldedLinear(MAX_REPEAT))).not.toThrow();
  });

  it('counts a folded parameter layer as N steps of depth', () => {
    const folded = foldedLinear(7);
    expect(analyze(folded).diagnostics.some(d => d.code === 'DEEP_NO_SKIP' && d.nodeId === 'head')).toBe(true);
    expect(analyze(foldedLinear()).diagnostics.some(d => d.code === 'DEEP_NO_SKIP')).toBe(false);
  });
});

describe('RMSNorm', () => {
  // RMSNorm scales by the root mean square only: no bias, no mean centering. It is a
  // shape-preserving op, so it also has to fold cleanly under repeat.
  const rmsGraph = (params: Params = {}, repeat?: number, shape: number[] = [2, 8]): Graph => {
    const graph = PRESETS.blank();
    graph.nodes[0].params.shape = shape;
    graph.nodes.splice(1, 0, { id: 'norm', name: 'Norm', op: 'RMSNorm', params: { ...params }, position: { x: 100, y: 100 }, ...(repeat === undefined ? {} : { repeat }) });
    graph.edges = [{ id: 'e0', source: 'layer_0', target: 'norm' }, { id: 'e1', source: 'norm', target: 'layer_1' }];
    return graph;
  };

  it('counts one weight per normalized element and no bias', () => {
    const analysis = analyze(rmsGraph({ normalized_shape: 8 }));
    expect(analysis.valid).toBe(true);
    expect(analysis.layers.norm.output).toEqual([2, 8]);
    expect(analysis.layers.norm.parameters).toBe(8);
    // LayerNorm owns weight and bias; RMSNorm owns weight only. Same shape, half the params.
    const layer = rmsGraph({ normalized_shape: 8 }); layer.nodes[1].op = 'LayerNorm';
    expect(analyze(layer).layers.norm.parameters).toBe(16);
  });

  it('drops the weight when elementwise_affine is off', () => {
    expect(analyze(rmsGraph({ normalized_shape: 8, elementwise_affine: 0 })).layers.norm.parameters).toBe(0);
    expect(analyze(rmsGraph({ normalized_shape: [2, 4] }, undefined, [3, 2, 4])).layers.norm.parameters).toBe(8);
  });

  it('is shape-preserving and validates normalized_shape against the input suffix', () => {
    expect(analyze(rmsGraph({ normalized_shape: [2, 8] }, undefined, [3, 2, 8])).layers.norm.output).toEqual([3, 2, 8]);
    // normalized_shape may not reach into the batch axis.
    expect(analyze(rmsGraph({ normalized_shape: [2, 8] })).valid).toBe(false);
    const mismatch = analyze(rmsGraph({ normalized_shape: 4 }));
    expect(mismatch.valid).toBe(false);
    expect(mismatch.diagnostics.some(d => d.nodeId === 'norm' && d.message.includes('RMSNorm normalized_shape 必须匹配'))).toBe(true);
    const illegal = analyze(rmsGraph({ normalized_shape: 0 }));
    expect(illegal.valid).toBe(false);
  });

  it('folds under repeat without changing shape and scales parameters by N', () => {
    const baseline = analyze(rmsGraph({ normalized_shape: 8 }));
    for (const repeat of [2, 3, 17, MAX_REPEAT]) {
      const analysis = analyze(rmsGraph({ normalized_shape: 8 }, repeat));
      expect(analysis.valid).toBe(true);
      expect(analysis.layers.norm.output).toEqual(baseline.layers.norm.output);
      expect(analysis.layers.norm.repeat).toBe(repeat);
      expect(analysis.layers.norm.parameters).toBe(8 * repeat);
      expect(analysis.parameters).toBe(8 * repeat);
    }
  });

  it('exports nn.RMSNorm with no bias and the analyzed parameter count', () => {
    const code = generatePython(rmsGraph({ normalized_shape: 8, eps: 1e-6 }));
    expect(code).toContain('nn.RMSNorm(8, eps=0.000001, elementwise_affine=True)');
    expect(code).not.toContain('nn.LayerNorm');
    expect(generatePython(rmsGraph({ normalized_shape: [2, 4], elementwise_affine: 0 }, undefined, [3, 2, 4]))).toContain('nn.RMSNorm((2, 4), eps=0.00001, elementwise_affine=False)');
  });

  it('execs the exported code and matches the backend analyzer', () => {
    for (const params of [{ normalized_shape: 8 }, { normalized_shape: 8, elementwise_affine: 0 }, { normalized_shape: [2, 4] }] as Params[]) {
      const shape = Array.isArray(params.normalized_shape) ? [3, 2, 4] : [3, 8];
      const graph = rmsGraph(params, undefined, shape), analysis = analyze(graph);
      expect(analysis.valid).toBe(true);
      const script = `import json, sys, torch
from backend.graph import analyze_graph
from backend.training import build_model
p = json.load(sys.stdin)
info = analyze_graph(p["graph"])
namespace = {"__name__": "rms_export"}
exec(p["code"], namespace)
exported = namespace["VisualModel"]()
backend_model, backend_info = build_model(p["graph"])
assert info["totalParameters"] == p["parameters"] == backend_info["totalParameters"]
assert sum(v.numel() for v in exported.parameters()) == p["parameters"]
x = torch.randn(*info["shapes"]["layer_0"])
with torch.no_grad():
    value = exported(x)
assert list(value.shape) == info["shapes"]["layer_1"], (value.shape, info["shapes"]["layer_1"])
print("matched", list(value.shape))
`;
      const result = execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify({ graph, code: generatePython(graph), parameters: analysis.parameters }), encoding: 'utf8', timeout: 30000 });
      expect(result).toContain('matched');
      expect(analysis.layers.layer_1.output).toEqual(shape);
    }
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
