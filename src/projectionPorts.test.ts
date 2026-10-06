import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze, validateGraph } from './analysis';
import { PRESETS } from './presets';
import { generatePython } from './export';

function routedGraph() {
  const graph = PRESETS.mqa();
  graph.nodes.push({ id: 'alternative', name: 'Alternative', op: 'Input', params: { shape: [1, 16, 64] }, position: { x: 0, y: 300 } });
  for (const port of ['q0', 'k0', 'v0']) graph.edges.push({ id: `override_${port}`, source: 'alternative', target: 'layer_1', targetPort: `b0:${port}` });
  graph.nodes.push({ id: 'projection', name: 'Q1 projection', op: 'Flatten', params: {}, position: { x: 420, y: 300 } });
  graph.edges.push({ id: 'q1_output', source: 'layer_1', sourcePort: 'b0:q0', target: 'projection' });
  graph.nodes.push({ id: 'concat', name: 'Joined', op: 'Concat', params: { dim: 1 }, position: { x: 550, y: 100 } });
  graph.edges = graph.edges.filter(e => e.target !== 'layer_3');
  graph.edges.push({ id: 'base_concat', source: 'layer_2', target: 'concat' }, { id: 'port_concat', source: 'projection', target: 'concat' }, { id: 'concat_linear', source: 'concat', target: 'layer_3' });
  return graph;
}
describe('Q/K/V input and output routing', () => {
  it('accepts separate ports between the same objects and propagates actual projection shapes', () => {
    const graph = routedGraph(), info = analyze(graph);
    expect(info.valid, JSON.stringify(info.diagnostics)).toBe(true);
    expect(validateGraph(graph)).toEqual(graph);
    expect(info.layers.projection.output).toEqual([1, 256]);
    expect(info.layers.concat.output).toEqual([1, 1280]);
  });
  it.each(['duplicate', 'missing', 'dimension', 'query_length', 'kv_length', 'cycle'])('rejects %s port routes before export and training', variant => {
    const graph = routedGraph();
    if (variant === 'duplicate') graph.edges.push({ id: 'duplicate', source: 'layer_0', target: 'layer_1', targetPort: 'b0:q0' });
    if (variant === 'missing') graph.edges.find(e => e.id === 'q1_output')!.sourcePort = 'b0:q9';
    if (variant === 'dimension') graph.nodes.find(n => n.id === 'alternative')!.params.shape = [1, 16, 32];
    if (variant === 'query_length') graph.nodes.find(n => n.id === 'alternative')!.params.shape = [1, 8, 64];
    if (variant === 'kv_length') graph.edges = graph.edges.filter(e => e.id !== 'override_v0');
    if (variant === 'kv_length') { graph.edges = graph.edges.filter(e => e.id !== 'override_q0'); graph.nodes.find(n => n.id === 'alternative')!.params.shape = [1, 8, 64]; }
    if (variant === 'cycle') graph.edges.push({ id: 'cycle', source: 'projection', target: 'layer_1', targetPort: 'b0:q1' });
    expect(analyze(graph).valid).toBe(false);
    expect(() => generatePython(graph)).toThrow();
    const script = 'import json,sys\nfrom backend.graph import analyze_graph\ntry:\n analyze_graph(json.load(sys.stdin))\nexcept ValueError:\n print("rejected")\nelse:\n raise AssertionError("Invalid port route accepted")';
    expect(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify(graph), encoding: 'utf8' })).toContain('rejected');
  });
  it('executes exported routes identically to backend, affects computation, and propagates gradients', () => {
    const graph = routedGraph(), code = generatePython(graph);
    const script = `import json,sys,torch\nfrom backend.training import build_model\np=json.load(sys.stdin)\nns={"__name__":"test"};exec(p["code"],ns)\nm,info=build_model(p["graph"]);e=ns["VisualModel"]();e.load_state_dict(m.state_dict());m.eval();e.eval()\nx={k:torch.randn(*info["shapes"][k],requires_grad=True) for k in info["inputs"]}\ny=m(x);assert torch.equal(y,e(x))\nassert sum(v.numel() for v in m.parameters())==p["parameters"]\ny.square().sum().backward();assert all(v.grad is not None and v.grad.abs().sum()>0 for v in x.values())\nchanged={**x,"alternative":x["alternative"]+2};assert not torch.allclose(y,m(changed))\nprint("matched")\n`;
    expect(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify({ graph, code, parameters: analyze(graph).parameters }), encoding: 'utf8', timeout: 30000 })).toContain('matched');
  }, 30000);
});
