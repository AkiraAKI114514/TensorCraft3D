import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze, validateGraph } from './analysis';
import { attentionConfig, incomingEdges } from './attentionConfig';
import { PRESETS } from './presets';
import { generatePython } from './export';

describe('attention architectures', () => {
  it('keeps legacy self-attention defaults without reducing KV heads', () => {
    const graph = PRESETS.transformer();
    delete graph.nodes[1].params.attention_type;
    delete graph.nodes[1].params.kv_heads;
    delete graph.nodes[1].params.branches;
    graph.nodes[1].params.num_heads = 4;
    expect(attentionConfig(graph.nodes[1].params)).toEqual({ type: 'self', heads: 4, kvHeads: 4, branches: 1 });
    expect(analyze(graph).layers.layer_1.parameters).toBe(33472);
    expect(analyze(graph).valid).toBe(true);
  });
  it('reduces real projection parameters for MQA/GQA and multiplies independent branches', () => {
    const mqa = PRESETS.mqa(), gqa = PRESETS.gqa();
    expect(analyze(mqa).layers.layer_1.parameters).toBe(10400);
    expect(analyze(gqa).layers.layer_1.parameters).toBe(27232);
    const original = analyze(mqa);
    mqa.nodes[1].params.branches = 3;
    expect(analyze(mqa).layers.layer_1.parameters).toBe(31200);
    expect(analyze(mqa).activationBytes).toBeGreaterThan(original.activationBytes);
    expect(generatePython(mqa)).toContain('attention_type="multi_query", branches=3');
  });
  it.each([
    { attention_type: 'unknown' }, { num_heads: 3 }, { kv_heads: 0 },
    { num_heads: 4, kv_heads: 3, attention_type: 'grouped_query' },
    { num_heads: 4, kv_heads: 8, attention_type: 'grouped_query' },
    { num_heads: 4, kv_heads: 2, attention_type: 'multi_query' },
    { num_heads: 4, kv_heads: 1, attention_type: 'self' },
    { branches: 0 }, { branches: 9 }, { branches: 1.5 },
    { attention_type: 'multi_branch', branches: 1 }
  ])('rejects invalid architecture parameters %j', params => {
    const graph = PRESETS.mqa(); Object.assign(graph.nodes[1].params, params);
    expect(analyze(graph).valid).toBe(false);
    expect(() => generatePython(graph)).toThrow();
  });
  it('resolves explicit Cross-Attention roles independently of edge order', () => {
    const graph = PRESETS.cross_attention(); graph.edges.reverse();
    expect(incomingEdges(graph, graph.nodes[1]).map(e => e.source)).toEqual(['layer_0', 'context']);
    const info = analyze(graph);
    expect(info.valid).toBe(true);
    expect(info.layers.layer_1.input).toEqual([[1, 8, 64], [1, 12, 64]]);
    expect(info.layers.layer_1.output).toEqual([1, 8, 64]);
    expect(generatePython(graph)).toContain('self.layers["layer_1"].forward_with_ports(values["layer_0"], values["context"], overrides={})');
    expect(validateGraph(graph)).toEqual(graph);
  });
  it('rejects missing/duplicate roles and invalid Context shape', () => {
    for (const variant of ['missing', 'duplicate', 'batch', 'embed', 'rank'] as const) {
      const graph = PRESETS.cross_attention();
      if (variant === 'missing') graph.edges.pop();
      if (variant === 'duplicate') graph.edges.at(-1)!.targetPort = 'query';
      if (variant === 'batch') graph.nodes.at(-1)!.params.shape = [2, 12, 64];
      if (variant === 'embed') graph.nodes.at(-1)!.params.shape = [1, 12, 32];
      if (variant === 'rank') graph.nodes.at(-1)!.params.shape = [1, 64];
      expect(analyze(graph).valid, variant).toBe(false);
    }
  });
  it('uses B × Q_heads × S_query × S_context × branches for attention limits', () => {
    const graph = PRESETS.cross_attention();
    graph.nodes[0].params.shape = [1, 512, 64]; graph.nodes.at(-1)!.params.shape = [1, 1024, 64];
    graph.nodes[1].params.branches = 8;
    expect(analyze(graph).diagnostics.some(d => d.message.includes('注意力矩阵'))).toBe(true);
  });
  it.each(['transformer', 'mqa', 'gqa', 'cross_attention'])('executes exported %s with backend-identical outputs, parameters and gradients', preset => {
    const graph = PRESETS[preset](); graph.nodes[1].params.branches = 2;
    const payload = { graph, code: generatePython(graph), parameters: analyze(graph).parameters };
    const script = `import json, sys, torch\nfrom backend.training import build_model\np = json.load(sys.stdin)\nnamespace = {"__name__": "export_test"}\nexec(p["code"], namespace)\nbackend, info = build_model(p["graph"])\nexported = namespace["VisualModel"]()\nexported.load_state_dict(backend.state_dict())\nbackend.eval(); exported.eval()\ninputs = {key: torch.randn(*info["shapes"][key], requires_grad=True) for key in info["inputs"]}\na, b = backend(inputs), exported(inputs)\nassert torch.equal(a, b), "Backend/export computation mismatch"\nassert sum(v.numel() for v in exported.parameters()) == p["parameters"] == info["totalParameters"]\nb.square().sum().backward()\nassert all(v.grad is not None and torch.isfinite(v.grad).all() for v in inputs.values())\nassert all(v.grad is not None and torch.isfinite(v.grad).all() for v in exported.parameters())\nprint("matched")\n`;
    expect(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify(payload), encoding: 'utf8', timeout: 30000 })).toContain('matched');
  }, 30000);
});
