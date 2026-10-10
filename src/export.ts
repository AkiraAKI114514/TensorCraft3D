import { analyze } from './analysis';
import { expandGraph, EXPAND_SEP } from './subgraph';
import { isGroup, repeatOf, type Graph, type Layer, type SubgraphDef } from './types';
import { attentionConfig, incomingEdges, isAttention, isProjectionPort } from './attentionConfig';
import attentionRuntime from '../backend/attention.py?raw';
import tensorRuntime from '../backend/tensor_ops.py?raw';

export function download(content: string | Blob, name: string, mime = 'text/plain') {
  const url = URL.createObjectURL(typeof content === 'string' ? new Blob([content], { type: mime }) : content);
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/** 子图定义的唯一入口和出口。任一条不成立就返回 null，调用方退回扁平展开。 */
function endpointsOf(def: SubgraphDef): { entry: Layer; exit: Layer } | null {
  const ids = new Set(def.nodes.map(n => n.id));
  if (!def.nodes.length) return null;
  if (def.edges.some(e => !ids.has(e.source) || !ids.has(e.target))) return null;
  // 块内不能再出现 Input/Output：它们描述整张图，不属于一个可复用的块。
  if (def.nodes.some(n => n.op === 'Input' || n.op === 'Output')) return null;
  const hasIncoming = new Set(def.edges.map(e => e.target));
  const hasOutgoing = new Set(def.edges.map(e => e.source));
  const entries = def.nodes.filter(n => !hasIncoming.has(n.id));
  const exits = def.nodes.filter(n => !hasOutgoing.has(n.id));
  if (entries.length !== 1 || exits.length !== 1) return null;
  return { entry: entries[0], exit: exits[0] };
}

export function generatePython(graph: Graph) {
  // 子图实例先展开：形状推导、参数计账和有效性检查都必须看到块内的真实算子。
  const expanded = expandGraph(graph);
  const analysis = analyze(expanded);
  if (!analysis.valid) throw new Error('请先修复形状或连线错误后导出代码');
  const modelInputs = expanded.nodes.filter(n => n.op === 'Input');
  const definitions: string[] = [];
  // 同名子图共享一个块类；入口形状不同则无法共享，该块退回扁平展开。
  const classMemo = new Map<string, { className: string; body: string }>();
  // 重复注意力同样共享一个包装类：它们的构造表达式逐字相同，所以只发射一份代码。
  const attentionClassMemo = new Map<string, string>();

  /**
   * 把一个块发射成可复用的 `nn.Module` 子类。
   *
   * 块必须是独立属性（`self.fc = nn.Linear(...)`），不能包成 `nn.ModuleDict`：导入器只对
   * 结构可证明一致的容器折叠，而 ModuleDict 一律不折。用独立属性和 `nn.ModuleList` 才能让
   * 「块 ×N」重新导入成一个块实例，而不是 N 份展开的算子。
   */
  const classFor = (def: SubgraphDef, instancePrefix: string, stack: string[]): string | null => {
    const name = def.id;
    if (stack.includes(name)) return null;
    const endpoints = endpointsOf(def);
    if (!endpoints) return null;
    // 块内节点只能消费块内产生的值：唯一允许的外部来源是入口那一条连线。
    // 入口不能按字面前缀推：入口本身可能是带 repeat 的嵌套块，展开后它的入口还要更深一层。
    // 因此直接在展开图里求「该实例子树中没有内部上游的那个节点」。
    const inside = new Set(expanded.nodes.filter(n => n.id.startsWith(instancePrefix)).map(n => n.id));
    if (!inside.size) return null;
    const internalTargets = new Set(expanded.edges.filter(e => inside.has(e.source) && inside.has(e.target)).map(e => e.target));
    const entries = [...inside].filter(id => !internalTargets.has(id));
    if (entries.length !== 1) return null;
    for (const edge of expanded.edges) {
      if (!inside.has(edge.target) || inside.has(edge.source)) continue;
      if (edge.target !== entries[0]) return null;
    }
    const className = `TensorLabBlock_${name.replace(/[^a-zA-Z0-9_]/g, '_')}`;
    const emitted = emitScope({ version: 1, name: def.name, nodes: def.nodes, edges: def.edges, subgraphs: graph.subgraphs }, instancePrefix, 'attributes', endpoints.entry.id, endpoints.exit.id, [...stack, name]);
    if (!emitted) return null;
    const body = `class ${className}(nn.Module):\n    def __init__(self):\n        super().__init__()\n${emitted.constructors.join('\n')}\n\n    def forward(self, x):\n        values = {}\n        ports = {}\n${emitted.forward.join('\n')}\n        return values[${JSON.stringify(emitted.exit)}]\n`;
    const cached = classMemo.get(name);
    // 同一个子图被多处实例化时，只有生成结果逐字相同才能共用一个类；形状不同说明它们
    // 并非同一个块，此时退回扁平展开而不是让其中一个用错参数。
    if (cached) return cached.body === body ? cached.className : null;
    definitions.push(body);
    classMemo.set(name, { className, body });
    return className;
  };

  /** 发射一层：顶层是 `ModuleDict`，块内是独立属性（见 classFor 的说明）。 */
  const emitScope = (level: Graph, prefix: string, mode: 'moduleDict' | 'attributes', entry: string | null, exit: string | null, stack: string[]): { constructors: string[]; forward: string[]; exit: string } | null => {
    const value = (edge: Graph['edges'][number]) => edge.sourcePort ? `ports[${JSON.stringify(edge.source)}][${JSON.stringify(edge.sourcePort)}]` : `values[${JSON.stringify(edge.source)}]`;
    const attr = (id: string) => mode === 'moduleDict' ? `self.layers[${JSON.stringify(id)}]` : `self.${id}`;
    const constructors: string[] = [], forward: string[] = [];
    let resolved = exit ?? '';
    const declare = (id: string, expr: string, repeat: number) => {
      if (repeat === 1) { constructors.push(mode === 'moduleDict' ? `            ${JSON.stringify(id)}: ${expr},` : `        self.${id} = ${expr}`); return; }
      const pad = mode === 'moduleDict' ? '                ' : '            ';
      const close = mode === 'moduleDict' ? '\n            ' : '\n        ';
      const items = Array.from({ length: repeat }, () => `\n${pad}${expr},`).join('');
      const list = `nn.ModuleList([${items}${close}])`;
      constructors.push(mode === 'moduleDict' ? `            ${JSON.stringify(id)}: ${list},` : `        self.${id} = ${list}`);
    };
    // 本层节点按展开图里的首次出现排序：`rank` 直接取「以 `<前缀><id>/` 或 `<前缀><id>` 开头的
    // 最小序号」，不把展开 id 再按顶层折叠一次——折叠会把带 `/` 的扁平回退节点错排。
    const rank = new Map<string, number>();
    for (const node of level.nodes) {
      const start = `${prefix}${node.id}`;
      for (const [index, expandedId] of analysis.order.entries()) {
        if (expandedId !== start && !expandedId.startsWith(`${start}${EXPAND_SEP}`)) continue;
        const current = rank.get(node.id);
        if (current === undefined || index < current) rank.set(node.id, index);
        break;
      }
    }
    const ordered = [...level.nodes].sort((a, b) => (rank.get(a.id) ?? Number.MAX_SAFE_INTEGER) - (rank.get(b.id) ?? Number.MAX_SAFE_INTEGER));

    for (const node of ordered) {
      const expandedId = `${prefix}${node.id}`;
      const p = node.params, info = analysis.layers[expandedId];
      const key = JSON.stringify(node.id);
      const edges = incomingEdges(level, node);
      const inputs = edges.filter(e => !isProjectionPort(e.targetPort)).map(value);
      if (entry === node.id && !inputs.length) inputs.push('x');

      if (isGroup(node)) {
        const definition = level.subgraphs?.[node.subgraph!];
        if (!definition) return null;
        // 实例前缀与 expandGraph 一致：`<实例 id>/<序号>/…`。
        const className = classFor(definition, `${expandedId}${EXPAND_SEP}0${EXPAND_SEP}`, stack);
        if (!className) return null;
        const repeat = repeatOf(node);
        declare(node.id, `${className}()`, repeat);
        const accessor = attr(node.id);
        if (repeat === 1) forward.push(`        values[${key}] = ${accessor}(${inputs[0]})`);
        else forward.push(`        values[${key}] = ${inputs[0]}\n        for instance in ${accessor}:\n            values[${key}] = instance(values[${key}])`);
        continue;
      }

      if (node.op === 'Input') { forward.push(`        values[${key}] = ${modelInputs.length === 1 && !prefix ? `x[${key}] if isinstance(x, dict) else x` : `x[${key}]`}`); continue; }
      if (node.op === 'Output') { forward.push(`        values[${key}] = ${inputs[0]}`); resolved = node.id; continue; }
      if (!info) return null;

      let expr = '';
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
        // nn.RMSNorm ships in torch >= 2.4 (the project pins >= 2.6), so the export needs no helper
        // class and stays trivially re-importable, exactly like nn.LayerNorm above. It has no bias.
        case 'RMSNorm': { const normalized = Array.isArray(p.normalized_shape) ? `(${p.normalized_shape.join(', ')})` : `${p.normalized_shape ?? info.input[0].at(-1)}`; expr = `nn.RMSNorm(${normalized}, eps=${p.eps ?? 1e-5}, elementwise_affine=${Number(p.elementwise_affine ?? 1) ? 'True' : 'False'})`; break; }
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
        case 'Multiply': forward.push(`        values[${key}] = ${inputs.join(' * ')}`); continue;
        case 'Concat': forward.push(`        values[${key}] = torch.cat([${inputs.join(', ')}], dim=${p.dim ?? 1})`); continue;
        // `torch.einsum` is native, so unlike ConstantAdd this needs no embedded runtime helper.
        case 'Einsum': forward.push(`        values[${key}] = torch.einsum(${JSON.stringify(p.equation)}, ${inputs.join(', ')})`); continue;
      }
      const repeat = repeatOf(node);
      const overrides = isAttention(node.op) ? edges.filter(e => isProjectionPort(e.targetPort)) : [];
      // A folded `Group` stands in for `count` identical instances, so the importer replays the
      // loop body once and resolves every index to that one node. The body therefore has to be a
      // plain single-input call; a port read or a projection override has no single node to hang
      // off, so an attention that uses either keeps the old, non-foldable shape.
      const portReaders = isAttention(node.op) && level.edges.some(e => e.source === node.id && e.sourcePort);
      let declared = expr;
      // repeat = N folds N isomorphic instances into one node. They are chained,
      // not weight-shared, so the export must materialise N distinct layers to
      // keep sum(p.numel()) equal to analysis.parameters. A single instance keeps
      // the historical `id: layer` shape so old exports stay byte-identical.
      //
      // A repeated non-attention op is a bare expression in a `nn.ModuleList`, and the importer
      // folds that. Attention is different: it is driven through `forward_with_ports`, and the
      // importer's Group branch only accepts a single positional argument, so a loop that calls
      // `instance.forward_with_ports(x, None, overrides=...)` is rejected outright. Wrap the
      // repeated attention in a one-operator class whose `forward` is the plain single-input
      // call; the class also exposes `forward_with_ports` so a downstream port consumer still
      // reads the same ports.
      // A repeated attention is only wrappable when it takes a single input. The wrapper's
      // `forward(self, x)` has one slot, so a Cross-Attention's Query/Context pair cannot go
      // through it — and the importer's Group branch accepts one positional argument anyway, so
      // such a call could never fold. Those keep the old `forward_with_ports` loop, which at
      // least still runs standalone; wrapping them silently dropped the context input.
      const wrapped = isAttention(node.op) && repeat > 1 && inputs.length === 1 && !overrides.length && !portReaders;
      if (wrapped) {
        let className = attentionClassMemo.get(expr);
        if (!className) {
          className = `TensorLabRepeatedAttention_${attentionClassMemo.size}`;
          attentionClassMemo.set(expr, className);
          definitions.push(`class ${className}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attention = ${expr}\n\n    def forward(self, x):\n        return self.attention(x)\n\n    def forward_with_ports(self, x, context=None, overrides=None):\n        return self.attention.forward_with_ports(x, context, overrides=overrides)\n`);
        }
        declared = `${className}()`;
      }
      declare(node.id, declared, repeat);
      const accessor = attr(node.id);
      if (isAttention(node.op)) {
        const overrideMap = `{${overrides.map(e => `${JSON.stringify(e.targetPort)}: ${value(e)}`).join(', ')}}`;
        if (repeat === 1) forward.push(`        values[${key}], ports[${key}] = ${accessor}.forward_with_ports(${inputs[0]}, ${inputs[1] ?? 'None'}, overrides=${overrideMap})`);
        else if (wrapped) forward.push(`        values[${key}] = ${inputs[0]}\n        for instance in ${accessor}:\n            values[${key}] = instance(values[${key}])`);
        else forward.push(`        values[${key}] = ${inputs[0]}\n        for instance in ${accessor}:\n            values[${key}], ports[${key}] = instance.forward_with_ports(values[${key}], ${inputs[1] ?? 'None'}, overrides=${overrideMap})`);
      } else if (repeat === 1) forward.push(`        values[${key}] = ${accessor}(${inputs.join(', ')})`);
      else forward.push(`        values[${key}] = ${inputs[0]}\n        for instance in ${accessor}:\n            values[${key}] = instance(values[${key}]${inputs.slice(1).map(i => `, ${i}`).join('')})`);
    }
    return { constructors, forward, exit: resolved };
  };

  // 优先把块发射成可复用类；任何一块不满足条件就整体退回扁平展开，保持既有行为。
  let emitted = emitScope(graph, '', 'moduleDict', null, null, []);
  if (!emitted) { definitions.length = 0; classMemo.clear(); attentionClassMemo.clear(); emitted = emitScope(expanded, '', 'moduleDict', null, null, []); }
  if (!emitted) throw new Error('请先修复形状或连线错误后导出代码');
  const helper = (expanded.nodes.some(n => isAttention(n.op)) ? `\n${attentionRuntime}\n` : '') + (expanded.nodes.some(n => n.op === 'ConstantAdd') ? `\n${tensorRuntime}\n` : '');
  const embeddingInputs = new Map<string, number>();
  for (const node of expanded.nodes.filter(n => n.op === 'Embedding')) {
    const edge = expanded.edges.find(e => e.target === node.id && !e.targetPort);
    if (edge && expanded.nodes.find(n => n.id === edge.source)?.op === 'Input') {
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
  const blocks = definitions.length ? `\n${definitions.join('\n')}\n` : '';
  return `"""Generated by TensorCraft3D. Image layout: NCHW; sequence layout: BSE."""\nimport torch\nfrom torch import nn\n${helper}${blocks}\n_TENSORLAB_INPUT_SHAPES = ${JSON.stringify(Object.fromEntries(modelInputs.map(n => [n.id, analysis.layers[n.id].output])))}\n\nclass VisualModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.ModuleDict({\n${emitted.constructors.join('\n')}\n        })\n\n    def forward(self, x):\n${inputCheck}        values = {}\n        ports = {}\n${emitted.forward.join('\n')}\n        return values[${JSON.stringify(emitted.exit)}]\n\n\nif __name__ == "__main__":\n    model = VisualModel().eval()\n    sample = ${sample}\n    with torch.no_grad():\n        result = model(sample)\n    print(model)\n    print("Output shape:", tuple(result.shape))\n    print("Parameters:", sum(p.numel() for p in model.parameters()))\n`;
}

export const safeFilename = (name: string) => name.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64) || 'model';
