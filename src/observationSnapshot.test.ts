import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { generatePython } from './export';
import { ObservationSnapshotError, parseObservationSnapshot, snapshotMatches, withChannel } from './observationSnapshot';
import { expandedLayerAt } from './subgraph';
import type { Graph, Layer } from './types';

const layer = (id: string, op: Layer['op'], params: Layer['params'] = {}, extra: Partial<Layer> = {}): Layer => ({ id, name: id, op, params, position: { x: 0, y: 0 }, ...extra });
const chain = (nodes: Layer[], extra: Partial<Graph> = {}): Graph => ({ version: 1, name: 'Observed', nodes, edges: nodes.slice(1).map((node, index) => ({ id: `e${index}`, source: nodes[index].id, target: node.id })), ...extra });

const cnn = () => chain([layer('in', 'Input', { shape: [2, 1, 6, 6] }), layer('conv', 'Conv2d', { out_channels: 3, kernel_size: 3, stride: 1, padding: 1, bias: 1 }), layer('act', 'ReLU'), layer('flat', 'Flatten'), layer('head', 'Linear', { out_features: 2 }), layer('out', 'Output')]);
/** A folded block: export writes it as a module list of block classes, the runtime expands it. */
const folded = () => chain([layer('in', 'Input', { shape: [2, 8] }), layer('stack', 'Group', {}, { subgraph: 'cell', repeat: 2 }), layer('head', 'Linear', { out_features: 2 }), layer('out', 'Output')], {
  subgraphs: { cell: { id: 'cell', name: 'Cell', origin: 'manual', nodes: [layer('fc', 'Linear', { out_features: 8 }), layer('act', 'Tanh')], edges: [{ id: 's0', source: 'fc', target: 'act' }] } },
});
const attention = () => chain([layer('in', 'Input', { shape: [1, 5, 8] }), layer('attn', 'MultiHeadAttention', { attention_type: 'self', embed_dim: 8, num_heads: 2, kv_heads: 2, branches: 1, dropout: 0, causal: 1 }), layer('flat', 'Flatten'), layer('head', 'Linear', { out_features: 2 }), layer('out', 'Output')]);

/**
 * What a user does outside the workbench: run the exported code, take the model as trained
 * (here: initialised from another seed), and snapshot one sample through the writer. The model's
 * own forward pass is the reference the snapshot has to reproduce; `probes` names submodules of
 * the exported model whose outputs are recorded as references too.
 */
function observe(graphs: { graph: Graph; sample: number[]; probes?: string[] }[]) {
  const script = `import json, sys, torch
from backend.observation_snapshot import capture_snapshot, snapshot_json
results = []
for case in json.load(sys.stdin):
    namespace = {"__name__": "observed_export"}
    exec(case["code"], namespace)
    torch.manual_seed(7)
    model = namespace["VisualModel"]().eval()
    sample = torch.randn(*case["sample"])
    probes = {}
    for path in case["probes"]:
        model.get_submodule(path).register_forward_hook(lambda module, inputs, output, path=path: probes.__setitem__(path, output.reshape(-1).tolist()))
    with torch.no_grad():
        expected = model(sample)
    results.append({"snapshot": snapshot_json(capture_snapshot(case["graph"], model, sample, label="exported model")), "expected": expected.reshape(-1).tolist(), "probes": probes})
print(json.dumps(results))
`;
  const input = JSON.stringify(graphs.map(({ graph, sample, probes = [] }) => ({ graph, sample, probes, code: generatePython(graph) })));
  return JSON.parse(execFileSync('.venv/Scripts/python.exe', ['-c', script], { input, encoding: 'utf8', timeout: 60000, maxBuffer: 64 * 1024 * 1024 })) as { snapshot: string; expected: number[]; probes: Record<string, number[]> }[];
}

describe('offline observation snapshots', () => {
  const [plain, block, attended] = observe([{ graph: cnn(), sample: [1, 1, 6, 6] }, { graph: folded(), sample: [1, 8], probes: ['layers.stack.0.fc', 'layers.stack.1.fc', 'layers.stack.1.act'] }, { graph: attention(), sample: [1, 5, 8] }]);

  it('reproduces the exported model\'s own output, folded blocks included', () => {
    for (const [result, graph] of [[plain, cnn()], [block, folded()]] as const) {
      const snapshot = parseObservationSnapshot(result.snapshot);
      expect(snapshot.weights).toBe('external');
      expect(snapshot.label).toBe('exported model');
      expect(snapshotMatches(graph, snapshot)).toBe(true);
      const measured = snapshot.tensors.find(tensor => tensor.nodeId === 'out')!;
      measured.slice.values.flat().forEach((value, index) => expect(value).toBeCloseTo(result.expected[index], 5));
    }
    // The runtime expands the block, so its layers are observed under their expanded IDs.
    expect(parseObservationSnapshot(block.snapshot).tensors.map(tensor => tensor.nodeId)).toEqual(['in', 'stack/0/fc', 'stack/0/act', 'stack/1/fc', 'stack/1/act', 'head', 'out']);
  });

  it('finds a layer of a folded block at the chosen instance, and the block itself at its exit', () => {
    const snapshot = parseObservationSnapshot(block.snapshot), graph = folded();
    // What the inspector shows offline: the drilled-in selection, mapped to its expanded ID.
    const observed = (drillPath: string[], selected: string, instances?: Record<string, number>) => {
      const target = expandedLayerAt(graph, drillPath, selected, instances);
      return target ? snapshot.tensors.find(tensor => tensor.nodeId === target.id) : undefined;
    };
    // Each instance owns its weights, so the two applications of `fc` really differ.
    expect(block.probes['layers.stack.0.fc']).not.toEqual(block.probes['layers.stack.1.fc']);
    for (const [tensor, nodeId, reference] of [
      [observed(['cell'], 'fc'), 'stack/0/fc', 'layers.stack.0.fc'],
      [observed(['cell'], 'fc', { cell: 1 }), 'stack/1/fc', 'layers.stack.1.fc'],
      [observed([], 'stack'), 'stack/1/act', 'layers.stack.1.act'],
    ] as const) {
      expect(tensor?.nodeId).toBe(nodeId);
      expect(tensor!.slice.truncated).toBe(false);
      tensor!.slice.values.flat().forEach((value, index) => expect(value).toBeCloseTo(block.probes[reference][index], 5));
    }
  });

  it('binds a folded block only to a graph that runs the same block the same number of times', () => {
    const snapshot = parseObservationSnapshot(block.snapshot);
    const repeated = folded(); repeated.nodes[1].repeat = 3;
    const changed = folded(); changed.subgraphs!.cell.nodes[1].op = 'ReLU';
    expect(snapshotMatches(repeated, snapshot)).toBe(false);
    expect(snapshotMatches(changed, snapshot)).toBe(false);
    const renamed = folded(); renamed.subgraphs!.cell.name = 'Renamed cell';
    expect(snapshotMatches(renamed, snapshot)).toBe(true);
  });

  it('carries the attention window and binds to the computation, not the layout', () => {
    const snapshot = parseObservationSnapshot(attended.snapshot);
    expect(snapshot.attentions).toHaveLength(1);
    expect(snapshot.attentions[0]).toMatchObject({ nodeId: 'attn', branch: 0, head: 0, mask: 'causal' });
    expect(snapshot.attentions[0].tensors.probabilities.slice.values[0].slice(1).every(value => value === 0)).toBe(true);
    const moved = attention(); moved.name = 'Renamed'; moved.nodes[1] = { ...moved.nodes[1], name: 'self-attention', position: { x: 300, y: 40 } };
    expect(snapshotMatches(moved, snapshot)).toBe(true);
    const changed = attention(); changed.nodes[1].params.causal = 0;
    expect(snapshotMatches(changed, snapshot)).toBe(false);
  });

  it('carries each layer\'s weights and checks them like any other tensor', () => {
    const snapshot = parseObservationSnapshot(plain.snapshot);
    expect(Object.keys(snapshot.parameters).sort()).toEqual(['conv', 'head']);
    expect(snapshot.parameters.conv.map(entry => [entry.name, entry.shape])).toEqual([['weight', [3, 1, 3, 3]], ['bias', [3]]]);
    const edit = (change: (value: Record<string, any>) => void) => { const value = JSON.parse(plain.snapshot) as Record<string, any>; change(value); return () => parseObservationSnapshot(JSON.stringify(value)); };
    expect(edit(value => { value.parameters.ghost = value.parameters.conv; })).toThrow('观测快照内容无效：parameters.ghost');
    expect(edit(value => { value.parameters.conv[0].nodeId = 'head'; })).toThrow('观测快照内容无效：parameters.conv[0].nodeId');
    expect(edit(value => { value.parameters.conv[0].owner = 'ghost'; })).toThrow('观测快照内容无效：parameters.conv[0].owner');
    // A file from before weights were written has none, and is still read.
    expect(parseObservationSnapshot(JSON.stringify({ ...JSON.parse(plain.snapshot), parameters: undefined })).parameters).toEqual({});
  });

  it('moves the main window only between channels it captured', () => {
    const conv = parseObservationSnapshot(plain.snapshot).tensors.find(tensor => tensor.nodeId === 'conv')!;
    expect(conv.channels).toMatchObject({ start: 0, count: 3, total: 3, selected: 0 });
    const third = withChannel(conv, 2);
    expect(third.slice.indices).toEqual([0, 2]);
    expect(third.slice.values).toBe(conv.channels!.maps[2]);
    expect(third.channels!.selected).toBe(2);
    // Statistics describe the whole tensor and do not move with the window.
    expect(third.stats).toBe(conv.stats);
    expect(withChannel(conv, 3)).toBe(conv);
    expect(withChannel(conv, 0)).toBe(conv);
  });

  it('refuses a file that is not a snapshot, or that does not match its own graph', () => {
    const reason = (edit: (value: Record<string, any>) => unknown) => {
      const value = JSON.parse(plain.snapshot) as Record<string, any>; edit(value);
      try { parseObservationSnapshot(JSON.stringify(value)); return 'accepted'; } catch (error) { expect(error).toBeInstanceOf(ObservationSnapshotError); return (error as Error).message; }
    };
    expect(() => parseObservationSnapshot('{')).toThrow('观测快照不是有效的 JSON');
    expect(reason(value => { value.format = 'tensorlab-project'; })).toBe('这不是 TensorCraft3D 观测快照');
    expect(reason(value => { value.version = 2; })).toBe('观测快照版本不受支持');
    expect(reason(value => { value.graph.version = 2; })).toBe('观测快照内容无效：graph');
    expect(reason(value => { value.tensors[0].nodeId = 'elsewhere'; })).toBe('观测快照内容无效：tensors[0].nodeId');
    expect(reason(value => { value.tensors.push(value.tensors[0]); })).toBe(`观测快照内容无效：tensors[${tensorCount(plain)}].nodeId`);
    expect(reason(value => { value.tensors[1].slice.values[0].push(0); })).toBe('观测快照内容无效：tensors[1].slice.values');
    expect(reason(value => { value.tensors[1].slice.values[0][0] = '1'; })).toBe('观测快照内容无效：tensors[1].slice.values[0][0]');
    expect(reason(value => { value.tensors[1].elements += 1; })).toBe('观测快照内容无效：tensors[1].elements');
    expect(reason(value => { value.tensors[1].channels.maps.pop(); })).toBe('观测快照内容无效：tensors[1].channels.maps');
    expect(reason(value => { value.tensors[1].channels.selected = 1; })).toBe('观测快照内容无效：tensors[1].channels');
    expect(reason(value => { value.weights = 'trained'; })).toBe('观测快照内容无效：weights');
    expect(reason(value => { value.label = 'x'.repeat(121); })).toBe('观测快照内容无效：label');
    const attentionFile = JSON.parse(attended.snapshot) as Record<string, any>;
    attentionFile.attentions[0].nodeId = 'head';
    expect(() => parseObservationSnapshot(JSON.stringify(attentionFile))).toThrow('观测快照内容无效：attentions[0].nodeId');
  });
});

function tensorCount(result: { snapshot: string }) {
  return (JSON.parse(result.snapshot) as { tensors: unknown[] }).tensors.length;
}
