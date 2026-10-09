import { analyze } from './analysis';
import { repeatOf, type Graph } from './types';
import { attentionConfig, incomingEdges, isAttention, isProjectionPort } from './attentionConfig';
import attentionRuntime from '../backend/attention.py?raw';
import tensorRuntime from '../backend/tensor_ops.py?raw';

export function download(content: string | Blob, name: string, mime = 'text/plain') {
  const url = URL.createObjectURL(typeof content === 'string' ? new Blob([content], { type: mime }) : content);
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function generatePython(graph: Graph) {
  const analysis = analyze(graph);
  if (!analysis.valid) throw new Error('请先修复形状或连线错误后导出代码');
  const constructors: string[] = [], forward: string[] = [];
  const modelInputs = graph.nodes.filter(n => n.op === 'Input');
  for (const id of analysis.order) {
    const node = graph.nodes.find(n => n.id === id)!;
    const p = node.params, info = analysis.layers[id];
    const value = (e: Graph['edges'][number]) => e.sourcePort ? `ports[${JSON.stringify(e.source)}][${JSON.stringify(e.sourcePort)}]` : `values[${JSON.stringify(e.source)}]`;
    const edges = incomingEdges(graph, node), inputs = edges.filter(e => !isProjectionPort(e.targetPort)).map(value);
    const key = JSON.stringify(id);
    let expr = '';
    if (node.op === 'Input') { forward.push(`        values[${key}] = ${modelInputs.length === 1 ? `x[${key}] if isinstance(x, dict) else x` : `x[${key}]`}`); continue; }
    if (node.op === 'Output') { forward.push(`        values[${key}] = ${inputs[0]}`); continue; }
    switch (node.op) {
      case 'Conv1d':
      case 'Conv2d':
      case 'Conv3d': expr = `nn.${node.op}(${info.input[0][1]}, ${p.out_channels ?? 16}, kernel_size=${p.kernel_size ?? 3}, stride=${p.stride ?? 1}, padding=${p.padding ?? 1}, groups=${p.groups ?? 1})`; break;
      case 'ConvTranspose1d':
      case 'ConvTranspose2d':
      case 'ConvTranspose3d': expr = `nn.${node.op}(${info.input[0][1]}, ${p.out_channels ?? 16}, kernel_size=${p.kernel_size ?? 4}, stride=${p.stride ?? 2}, padding=${p.padding ?? 1}, output_padding=${p.output_padding ?? 0}, groups=${p.groups ?? 1})`; break;
      case 'Linear': expr = `nn.Linear(${info.input[0].at(-1)}, ${p.out_features ?? 10})`; break;
      case 'Bilinear': expr = `nn.Bilinear(${info.input[0][1]}, ${p.in2_features ?? 16}, ${p.out_features ?? 10})`; break;
      case 'BatchNorm1d':
      case 'BatchNorm2d':
      case 'BatchNorm3d':
      case 'InstanceNorm1d':
      case 'InstanceNorm2d':
      case 'InstanceNorm3d': expr = `nn.${node.op}(${info.input[0][1]}${node.op.startsWith('InstanceNorm') ? `, affine=${Number(p.affine ?? 1) ? 'True' : 'False'}, track_running_stats=${Number(p.track_running_stats ?? 0) ? 'True' : 'False'}` : ''})`; break;
      case 'LayerNorm': { const normalized = Array.isArray(p.normalized_shape) ? `(${p.normalized_shape.join(', ')})` : `${p.normalized_shape ?? info.input[0].at(-1)}`; expr = `nn.LayerNorm(${normalized}, eps=${p.eps ?? 1e-5}, elementwise_affine=${Number(p.elementwise_affine ?? 1) ? 'True' : 'False'})`; break; }
      case 'GroupNorm': expr = `nn.GroupNorm(${p.num_groups ?? 1}, ${info.input[0][1]}, affine=${Number(p.affine ?? 1) ? 'True' : 'False'})`; break;
      case 'MaxPool1d':
      case 'MaxPool2d':
      case 'MaxPool3d':
      case 'AvgPool1d':
      case 'AvgPool2d':
      case 'AvgPool3d': expr = `nn.${node.op}(kernel_size=${p.kernel_size ?? 2}, stride=${p.stride ?? 2}, padding=${p.padding ?? 0})`; break;
      case 'AdaptiveAvgPool1d':
      case 'AdaptiveAvgPool2d':
      case 'AdaptiveAvgPool3d':
      case 'AdaptiveMaxPool1d':
      case 'AdaptiveMaxPool2d':
      case 'AdaptiveMaxPool3d': expr = `nn.${node.op}(${Array.isArray(p.output_size) ? `(${p.output_size.join(', ')})` : p.output_size ?? 1})`; break;
      case 'Flatten': expr = 'nn.Flatten(start_dim=1)'; break;
      case 'Unsqueeze': forward.push(`        values[${key}] = torch.unsqueeze(${inputs[0]}, dim=${p.dim ?? 0})`); continue;
      case 'Squeeze': {
        const dim = p.dim ?? 'all', argument = dim === 'all' ? '' : `, dim=${Array.isArray(dim) ? `(${dim.join(', ')}${dim.length === 1 ? ',' : ''})` : dim}`;
        forward.push(`        values[${key}] = torch.squeeze(${inputs[0]}${argument})`); continue;
      }
      case 'Slice': {
        const dim = (Number(p.dim ?? 1) + info.input[0].length) % info.input[0].length;
        const bound = (name: string) => p[name] === undefined || p[name] === 'none' ? '' : String(p[name]);
        const indices = Array.from({ length: dim + 1 }, (_, i) => i === dim ? `${bound('start')}:${bound('end')}:${p.step ?? 1}` : ':');
        forward.push(`        values[${key}] = ${inputs[0]}[${indices.join(', ')}]`); continue;
      }
      case 'Select': forward.push(`        values[${key}] = torch.select(${inputs[0]}, dim=${p.dim ?? 1}, index=${p.index ?? -1})`); continue;
      case 'ConstantAdd': expr = `TensorLabConstantAdd(shape=${JSON.stringify(p.shape)}, values=${JSON.stringify(p.values)}${p.sequence_dim === undefined ? '' : `, sequence_dim=${p.sequence_dim}`})`; break;
      case 'Dropout':
      case 'Dropout1d':
      case 'Dropout2d':
      case 'Dropout3d':
      case 'AlphaDropout': expr = `nn.${node.op}(p=${p.p ?? 0.3})`; break;
      case 'Embedding': expr = `nn.Embedding(${p.num_embeddings ?? 100}, ${p.embedding_dim ?? 32})`; break;
      case 'Upsample': expr = `nn.Upsample(scale_factor=${Array.isArray(p.scale_factor) ? `(${p.scale_factor.join(', ')})` : p.scale_factor ?? 2}, mode=${JSON.stringify(p.mode ?? 'nearest')})`; break;
      case 'ReLU':
      case 'GELU':
      case 'Sigmoid':
      case 'Tanh':
      case 'SiLU':
      case 'SELU':
      case 'Hardsigmoid':
      case 'Hardswish':
      case 'Mish':
      case 'Softsign':
      case 'Identity': expr = `nn.${node.op}()`; break;
      case 'PReLU': expr = `nn.PReLU(num_parameters=${p.num_parameters ?? 1}, init=${p.init ?? 0.25})`; break;
      case 'LeakyReLU': expr = `nn.LeakyReLU(negative_slope=${p.negative_slope ?? 0.01})`; break;
      case 'ELU': expr = `nn.ELU(alpha=${p.alpha ?? 1})`; break;
      case 'Softplus': expr = `nn.Softplus(beta=${p.beta ?? 1}, threshold=${p.threshold ?? 20})`; break;
      case 'Softmax':
      case 'LogSoftmax': expr = `nn.${node.op}(dim=${p.dim ?? -1})`; break;
      case 'MultiHeadAttention':
      case 'Transformer': {
        const cfg = attentionConfig(p);
        expr = `${node.op === 'Transformer' ? 'TensorLabTransformer' : 'TensorLabAttention'}(embed_dim=${p.embed_dim ?? 64}, num_heads=${cfg.heads}, kv_heads=${cfg.kvHeads}, dropout=${p.dropout ?? 0.1}, attention_type=${JSON.stringify(cfg.type)}, branches=${cfg.branches}${node.op === 'Transformer' ? `, ff_dim=${p.ff_dim ?? 128}, norm_first=${Number(p.norm_first ?? 1) ? 'True' : 'False'}, activation=${JSON.stringify(p.activation ?? 'gelu')}` : ''})`;
        break;
      }
      case 'Add': forward.push(`        values[${key}] = ${inputs.join(' + ')}`); continue;
      case 'Concat': forward.push(`        values[${key}] = torch.cat([${inputs.join(', ')}], dim=${p.dim ?? 1})`); continue;
    }
    // repeat = N folds N isomorphic instances into one node. They are chained,
    // not weight-shared, so the export must materialise N distinct layers to
    // keep sum(p.numel()) equal to analysis.parameters. A single instance keeps
    // the historical `id: layer` shape so old exports stay byte-identical.
    const repeat = repeatOf(node);
    if (repeat === 1) constructors.push(`            ${key}: ${expr},`);
    else constructors.push(`            ${key}: nn.ModuleList([${Array.from({ length: repeat }, () => `\n                ${expr},`).join('')}\n            ]),`);
    if (isAttention(node.op)) {
      const overrides = `{${edges.filter(e => isProjectionPort(e.targetPort)).map(e => `${JSON.stringify(e.targetPort)}: ${value(e)}`).join(', ')}}`;
      if (repeat === 1) forward.push(`        values[${key}], ports[${key}] = self.layers[${key}].forward_with_ports(${inputs[0]}, ${inputs[1] ?? 'None'}, overrides=${overrides})`);
      else forward.push(`        values[${key}] = ${value(edges.filter(e => !isProjectionPort(e.targetPort))[0])}\n        for instance in self.layers[${key}]:\n            values[${key}], ports[${key}] = instance.forward_with_ports(values[${key}], ${inputs[1] ?? 'None'}, overrides=${overrides})`);
    } else if (repeat === 1) forward.push(`        values[${key}] = self.layers[${key}](${inputs.join(', ')})`);
    else forward.push(`        values[${key}] = ${inputs[0]}\n        for instance in self.layers[${key}]:\n            values[${key}] = instance(values[${key}]${inputs.slice(1).map(i => `, ${i}`).join('')})`);
  }
  const output = graph.nodes.find(n => n.op === 'Output')!;
  const helper = (graph.nodes.some(n => isAttention(n.op)) ? `\n${attentionRuntime}\n` : '') + (graph.nodes.some(n => n.op === 'ConstantAdd') ? `\n${tensorRuntime}\n` : '');
  const embeddingInputs = new Map<string, number>();
  for (const node of graph.nodes.filter(n => n.op === 'Embedding')) {
    const edge = graph.edges.find(e => e.target === node.id && !e.targetPort);
    if (edge && graph.nodes.find(n => n.id === edge.source)?.op === 'Input') {
      embeddingInputs.set(edge.source, Number(node.params.num_embeddings ?? 100));
    }
  }
  const sampleFor = (node: Graph['nodes'][number]) => {
    const shape = analysis.layers[node.id].output.join(', ');
    const entries = embeddingInputs.get(node.id);
    return entries ? `torch.randint(0, ${entries}, (${shape}), dtype=torch.long)` : `torch.randn(${shape})`;
  };
  const sample = modelInputs.length === 1 ? sampleFor(modelInputs[0]) : `{\n${modelInputs.map(n => `        ${JSON.stringify(n.id)}: ${sampleFor(n)},`).join('\n')}\n    }`;
  const inputCheck = `        input_ids = ${JSON.stringify(modelInputs.map(n => n.id))}\n        if isinstance(x, dict):\n            if set(x) != set(input_ids):\n                raise ValueError("Input dictionary must contain exactly the model's Input node IDs")\n        elif len(input_ids) != 1:\n            raise ValueError("Multi-input models require a dictionary keyed by Input node ID")\n`;
  return `"""Generated by TensorCraft3D. Image layout: NCHW; sequence layout: BSE."""\nimport torch\nfrom torch import nn\n${helper}\n\n_TENSORLAB_INPUT_SHAPES = ${JSON.stringify(Object.fromEntries(modelInputs.map(n => [n.id, analysis.layers[n.id].output])))}\n\nclass VisualModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleDict({\n${constructors.join('\n')}\n        })\n\n    def forward(self, x):\n${inputCheck}        values = {}\n        ports = {}\n${forward.join('\n')}\n        return values[${JSON.stringify(output.id)}]\n\n\nif __name__ == "__main__":\n    model = VisualModel().eval()\n    sample = ${sample}\n    with torch.no_grad():\n        result = model(sample)\n    print(model)\n    print("Output shape:", tuple(result.shape))\n    print("Parameters:", sum(p.numel() for p in model.parameters()))\n`;
}

export const safeFilename = (name: string) => name.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64) || 'model';
