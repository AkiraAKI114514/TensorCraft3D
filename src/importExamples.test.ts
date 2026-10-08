import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { analyze, validateGraph } from './analysis';
import { generatePython } from './export';
import type { Graph } from './types';

type Example = { filename: string; model_name: string; input_shapes: Record<string, number[]>; expected_output: number[]; expected_parameters: number };
const { examples } = JSON.parse(readFileSync(new URL('../examples/import_models/manifest.json', import.meta.url), 'utf8')) as { examples: Example[] };

describe('reproducible source import and standalone export', () => {
  it.each(examples)('$filename preserves graph/export/roundtrip values and gradients', example => {
    const source = readFileSync(new URL(`../examples/import_models/${example.filename}`, import.meta.url), 'utf8');
    const parse = `import json,sys
from backend.pytorch_import import import_pytorch
p=json.load(sys.stdin)
r=import_pytorch(p['source'],p['model_name'],p['input_shapes'])
assert r['graph'] is not None,r['diagnostics']
print(json.dumps(r['graph']))
`;
    const graph = JSON.parse(execFileSync('.venv/Scripts/python.exe', ['-c', parse], { input: JSON.stringify({ ...example, source }), encoding: 'utf8', timeout: 30000 })) as Graph;
    const validated = validateGraph(graph), analysis = analyze(validated);
    expect(analysis.valid).toBe(true);
    expect(analysis.parameters).toBe(example.expected_parameters);
    const code = generatePython(validated);
    const script = `import ast,json,sys,torch
from backend.training import build_model
from backend.pytorch_import import import_pytorch
p=json.load(sys.stdin)
backend,info=build_model(p['graph'])
namespace={'__name__':'export_fixture'}
exec(p['code'],namespace)
exported=namespace['VisualModel']()
exported.load_state_dict(backend.state_dict(),strict=True)
r=import_pytorch(p['code'])
assert r['graph'] is not None,r['diagnostics']
roundtrip,roundtrip_info=build_model(r['graph'])
for node in r['graph']['nodes']:
 if node['id'] in roundtrip.layers:
  key=ast.literal_eval(node['name'].split('self.layers',1)[1][1:-1])
  roundtrip.layers[node['id']].load_state_dict(backend.layers[key].state_dict())
backend.eval();exported.eval();roundtrip.eval()
assert info['totalParameters']==roundtrip_info['totalParameters']==p['expected_parameters']
x={key:torch.randn(*info['shapes'][key],requires_grad=True) for key in info['inputs']}
y={key:value.detach().clone().requires_grad_() for key,value in x.items()}
z={item['id']:x[item['name']].detach().clone().requires_grad_() for item in r['inputs']}
a,b,c=backend(x),exported(y),roundtrip(z)
assert list(a.shape)==p['expected_output']
torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
torch.testing.assert_close(a,c,atol=1e-6,rtol=1e-5)
a.square().sum().backward();b.square().sum().backward();c.square().sum().backward()
for key in x:
 torch.testing.assert_close(x[key].grad,y[key].grad,atol=1e-6,rtol=1e-4)
for item in r['inputs']:
 torch.testing.assert_close(x[item['name']].grad,z[item['id']].grad,atol=1e-6,rtol=1e-4)
print('matched')
`;
    expect(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input: JSON.stringify({ ...example, graph: validated, code }), encoding: 'utf8', timeout: 30000 })).toContain('matched');
  }, 60000);
});
