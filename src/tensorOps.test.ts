import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { analyze, validateGraph } from './analysis';
import { generatePython } from './export';
import type { Graph, Params } from './types';

function graph(): Graph {
  return {
    version: 1, name: 'Position and slicing',
    nodes: [
      { id: 'input', name: 'Input', op: 'Input', params: { shape: [2, 6, 4] }, position: { x: 0, y: 0 } },
      { id: 'position', name: 'Position', op: 'ConstantAdd', params: { shape: [1, 8, 4], values: Array.from({ length: 32 }, (_, i) => i / 10), sequence_dim: 1 }, position: { x: 200, y: 0 } },
      { id: 'slice', name: 'Slice', op: 'Slice', params: { dim: 1, start: 1, end: 'none', step: 2 }, position: { x: 400, y: 0 } },
      { id: 'last', name: 'Last step', op: 'Select', params: { dim: 1, index: -1 }, position: { x: 600, y: 0 } },
      { id: 'head', name: 'Head', op: 'Linear', params: { out_features: 2 }, position: { x: 800, y: 0 } },
      { id: 'output', name: 'Output', op: 'Output', params: {}, position: { x: 1000, y: 0 } },
    ],
    edges: ['input', 'position', 'slice', 'last', 'head'].map((source, i) => ({ id: `e${i}`, source, target: ['position', 'slice', 'last', 'head', 'output'][i] })),
  };
}

describe('fixed buffers and tensor indexing contracts', () => {
  it('keeps constant values in projects and derives actual sliced shapes', () => {
    const model = graph();
    expect(validateGraph(model)).toEqual(model);
    const info = analyze(model);
    expect(info.valid, JSON.stringify(info.diagnostics)).toBe(true);
    expect(info.layers.slice.output).toEqual([2, 3, 4]);
    expect(info.layers.last.output).toEqual([2, 4]);
    expect(info.layers.output.output).toEqual([2, 2]);
    expect(info.parameters).toBe(10);
  });

  it.each([
    ['position', { values: [1] }], ['position', { sequence_dim: 0 }], ['position', { shape: [2, 4], values: Array(8).fill(0) }],
    ['position', { shape: [1, 5, 4], values: Array(20).fill(0) }], ['position', { values: Array(32).fill(1e39) }],
    ['slice', { dim: 0 }], ['slice', { step: 0 }], ['slice', { step: -1 }], ['slice', { start: 5, end: 1 }], ['last', { index: 10 }],
  ] as [string, Params][])('rejects invalid %s parameters %j before export', (id, params) => {
    const model = graph(), node = model.nodes.find(n => n.id === id)!;
    node.params = { ...node.params, ...params };
    expect(analyze(model).valid).toBe(false);
    expect(() => generatePython(model)).toThrow();
  });

  it('allows bounded buffer arrays without widening other parameter arrays', () => {
    const model = graph();
    model.nodes[1].params = { shape: [1, 16, 4], values: Array(64).fill(0), sequence_dim: 1 };
    expect(validateGraph(model)).toEqual(model);
    model.nodes[0].params.shape = Array(9).fill(1);
    expect(() => validateGraph(model)).toThrow();
    model.nodes[0].params.shape = [2, 6, 4];
    model.nodes[1].params.values = Array(65537).fill(0);
    expect(() => validateGraph(model)).toThrow();
  });

  it('matches backend, standalone Python and reimport with dynamic batch and sequence lengths', () => {
    const model = graph(), code = generatePython(model);
    const script = `import ast,json,sys,torch
from backend.training import build_model
from backend.pytorch_import import import_pytorch
p=json.load(sys.stdin)
a,info=build_model(p['graph'])
ns={'__name__':'test_export'};exec(p['code'],ns)
b=ns['VisualModel']();b.load_state_dict(a.state_dict())
r=import_pytorch(p['code']);assert r['graph'] is not None,r['diagnostics']
c,ci=build_model(r['graph'])
for n in r['graph']['nodes']:
 if n['id'] in c.layers:
  old=ast.literal_eval(n['name'].split('self.layers',1)[1][1:-1])
  c.layers[n['id']].load_state_dict(a.layers[old].state_dict())
assert ci['totalParameters']==info['totalParameters']==p['parameters']
for batch,length in ((1,4),(3,6),(2,8)):
 x=torch.randn(batch,length,4,requires_grad=True)
 y,z,w=a(x),b(x),c(x)
 assert torch.equal(y,z) and torch.equal(y,w)
 expected=a.layers['head']((x+a.layers['position'].constant[:,:length])[:,1::2][:,-1])
 torch.testing.assert_close(y,expected)
 torch.testing.assert_close(torch.autograd.grad(z.square().sum(),x)[0],torch.autograd.grad(expected.square().sum(),x)[0])
print(json.dumps(r['graph']))
`;
    const imported = JSON.parse(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify({ graph: model, code, parameters: analyze(model).parameters }), encoding: 'utf8', timeout: 30000 })) as Graph;
    expect(validateGraph(imported)).toEqual(imported);
    expect(analyze(imported).valid).toBe(true);
    expect(imported.nodes.filter(n => ['ConstantAdd', 'Slice', 'Select'].includes(n.op))).toHaveLength(3);
  }, 30000);
});
