import { OPS, type Graph, type Analysis, type LayerInfo, type Diagnostic, type Metric } from './types';
import { ATTENTION_TYPES, attentionConfig, incomingEdges, isCrossAttention, crossInputRole, edgeKey, edgeOutputShape, isAttention, isProjectionPort, projectionPorts } from './attentionConfig';

export const product = (values: number[]) => values.reduce((a, b) => a * b, 1);
export const shapeText = (shape?: number[]) => shape ? shape.join(' × ') : '—';
export const formatNumber = (value: number) => value >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}K` : String(value);

export function validateGraph(value: unknown): Graph {
  const g = value as Graph;
  if (!g || g.version !== 1 || typeof g.name !== 'string' || !Array.isArray(g.nodes) || !Array.isArray(g.edges)) throw new Error('不是有效的 TensorLab v1 项目');
  if (g.name.length > 120 || g.nodes.length > 128 || g.edges.length > 512) throw new Error('项目超过限制（128 层 / 512 连线）');
  const ids = new Set<string>();
  g.nodes.forEach(n => {
    if (!n || typeof n.id !== 'string' || !/^[a-zA-Z][a-zA-Z0-9_]{0,63}$/.test(n.id) || ids.has(n.id) || !OPS.includes(n.op) || typeof n.name !== 'string' || n.name.length > 120 || !n.params || typeof n.params !== 'object' || Array.isArray(n.params)) throw new Error('节点数据或 ID 无效');
    if (!n.position || !Number.isFinite(n.position.x) || !Number.isFinite(n.position.y)) throw new Error('节点坐标无效');
    for (const [key, v] of Object.entries(n.params)) if (!(typeof v === 'number' && Number.isFinite(v)) && !(typeof v === 'string' && v.length < 200) && !(Array.isArray(v) && v.length <= (n.op === 'ConstantAdd' && key === 'values' ? 65536 : 8) && v.every(x => typeof x === 'number' && Number.isFinite(x)))) throw new Error('节点参数无效');
    ids.add(n.id);
  });
  const edgeIds = new Set<string>(), pairs = new Set<string>();
  g.edges.forEach(e => {
    const pair = edgeKey(e);
    if (typeof e.id !== 'string' || edgeIds.has(e.id) || pairs.has(pair) || !ids.has(e.source) || !ids.has(e.target) || e.source === e.target || (e.targetPort !== undefined && !['query', 'context'].includes(e.targetPort) && !isProjectionPort(e.targetPort)) || (e.sourcePort !== undefined && !isProjectionPort(e.sourcePort))) throw new Error('连线无效或重复');
    edgeIds.add(e.id); pairs.add(pair);
  });
  return structuredClone(g);
}

export function analyze(graph: Graph): Analysis {
  const layers: Record<string, LayerInfo> = {}, diagnostics: Diagnostic[] = [], order: string[] = [];
  const add = (code: string, message: string, nodeId?: string, level: Diagnostic['level'] = 'error') => diagnostics.push({ id: `${code}_${nodeId || diagnostics.length}`, code, message, nodeId, level });
  const indegree = new Map(graph.nodes.map(n => [n.id, graph.edges.filter(e => e.target === n.id).length]));
  const queue = graph.nodes.filter(n => indegree.get(n.id) === 0).map(n => n.id);
  while (queue.length) {
    const id = queue.shift()!; order.push(id);
    graph.edges.filter(e => e.source === id).forEach(e => { indegree.set(e.target, indegree.get(e.target)! - 1); if (indegree.get(e.target) === 0) queue.push(e.target); });
  }
  if (order.length !== graph.nodes.length) add('CYCLE', '计算图存在循环，请移除形成回路的连线');
  const inputCount = graph.nodes.filter(n => n.op === 'Input').length;
  if (inputCount < 1 || inputCount > 8) add('INPUT', '项目需要 1–8 个输入层，多输入以节点 ID 区分');
  if (graph.nodes.filter(n => n.op === 'Output').length !== 1) add('OUTPUT', '项目需要且只能包含一个输出层');
  const outputNode = graph.nodes.find(n => n.op === 'Output');
  const ancestors = new Set<string>();
  const visit = (id: string) => { if (ancestors.has(id)) return; ancestors.add(id); graph.edges.filter(e => e.target === id).forEach(e => visit(e.source)); };
  if (outputNode) visit(outputNode.id);
  let parameters = 0, activationBytes = 0, constantElements = 0;
  const depth: Record<string, number> = {};
  for (const id of order) {
    const n = graph.nodes.find(n => n.id === id)!;
    const allEdges = incomingEdges(graph, n), inputEdges = allEdges.filter(e => !isProjectionPort(e.targetPort)), parents = inputEdges.map(e => e.source);
    const input = inputEdges.map(e => edgeOutputShape(graph, e, layers)).filter((s): s is number[] => Boolean(s));
    try {
      const integer = (key: string, fallback: number, min = 1, max = 65536) => { const v = n.params[key] ?? fallback; if (typeof v !== 'number' || !Number.isInteger(v) || v < min || v > max) throw new Error(`${key} 必须是 ${min}–${max} 的整数`); return v; };
      for (const edge of allEdges) {
        if (edge.sourcePort && !projectionPorts(graph.nodes.find(node => node.id === edge.source)!).some(p => p.id === edge.sourcePort)) throw new Error(`输出端口 ${edge.sourcePort} 不存在`);
        if (isProjectionPort(edge.targetPort) && !projectionPorts(n).some(p => p.id === edge.targetPort)) throw new Error(`输入端口 ${edge.targetPort} 不存在`);
        if (!edgeOutputShape(graph, edge, layers)) throw new Error('缺少有效输入，请检查上游层和连线');
      }
      if (n.op === 'Input' && parents.length) throw new Error('输入层不能有上游连线');
      if (n.op !== 'Input' && (!parents.length || input.length !== parents.length)) throw new Error('缺少有效输入，请检查上游层和连线');
      if (isCrossAttention(n)) {
        if (parents.length !== 2 || new Set(inputEdges.map(e => crossInputRole(graph, n, e.id))).size !== 2) throw new Error('Cross-Attention 需要两个输入：一个 Query 和一个 Context');
      } else {
        if (inputEdges.some(e => e.targetPort)) throw new Error('Query / Context 端口只适用于 Cross-Attention');
        if (!['Input', 'Add', 'Concat', 'Bilinear'].includes(n.op) && parents.length !== 1) throw new Error('该层只能接受一个输入；多分支请使用 Add 或 Concat');
        if (n.op === 'Bilinear' && parents.length !== 2) throw new Error('Bilinear 需要两个输入');
      }
      let output = [...(input[0] || [])], count = 0;
      if (n.op === 'Input') {
        const s = n.params.shape;
        if (!Array.isArray(s) || ![2, 3, 4, 5].includes(s.length) || !s.every(v => Number.isInteger(v) && v > 0 && v <= 65536) || product(s) > 16e6) throw new Error('输入形状须为 [B,F]、[B,S,E]、[B,C,H,W] 或 [B,C,D,H,W]，元素数不超过 1600 万');
        output = [...s];
      } else if (['ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'].includes(n.op)) {
        const rank = Number(n.op.match(/\d$/)?.[0] || 2) + 2;
        if (output.length !== rank) throw new Error(`${n.op} 需要 ${rank} 维通道输入`);
        const k = integer('kernel_size', 4, 1, 64), stride = integer('stride', 2, 1, 64), padding = integer('padding', 1, 0, 64), outputPadding = integer('output_padding', 0, 0, 64);
        if (outputPadding >= stride) throw new Error('output_padding 必须小于 stride');
        const channels = integer('out_channels', 16, 1, 4096), groups = integer('groups', 1, 1, output[1]);
        if (output[1] % groups || channels % groups) throw new Error('channels 必须能被 groups 整除');
        const spatial = output.slice(2).map(size => (size - 1) * stride - 2 * padding + k + outputPadding);
        if (spatial.some(v => v <= 0)) throw new Error('转置卷积输出尺寸无效');
        count = (output[1] * (channels / groups) * Math.pow(k, rank - 2)) + channels;
        output = [output[0], channels, ...spatial];
      } else if (['Conv1d', 'Conv2d', 'Conv3d', 'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d'].includes(n.op)) {
        const rank = Number(n.op.match(/\d$/)?.[0] || 2) + 2;
        if (output.length !== rank) throw new Error(`${n.op} 需要 ${rank} 维通道输入`);
        const convolution = n.op.startsWith('Conv'), k = integer('kernel_size', convolution ? 3 : 2, 1, 64), stride = integer('stride', convolution ? 1 : 2, 1, 64), padding = integer('padding', convolution ? 1 : 0, 0, 64);
        const spatial = output.slice(2).map(size => Math.floor((size + 2 * padding - k) / stride) + 1);
        if (spatial.some(v => v <= 0)) throw new Error('卷积核或池化核大于输入，输出尺寸为零');
        const channels = convolution ? integer('out_channels', 16, 1, 4096) : output[1];
        if (convolution) { const groups = integer('groups', 1, 1, output[1]); if (output[1] % groups || channels % groups) throw new Error('channels 必须能被 groups 整除'); count = channels * (output[1] / groups) * Math.pow(k, rank - 2) + channels; }
        output = [output[0], channels, ...spatial];
      } else if (['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d'].includes(n.op)) {
        const rank = Number(n.op.match(/\d$/)?.[0] || 2) + 2; if (output.length !== rank) throw new Error(`${n.op} 需要 ${rank} 维通道输入`); if (n.op.startsWith('BatchNorm') || Number(n.params.affine ?? 1)) count = output[1] * 2;
      } else if (n.op === 'LayerNorm') {
        const raw = n.params.normalized_shape, normalized: number[] = (Array.isArray(raw) ? raw : [raw]).map(Number);
        if (!normalized.length || !normalized.every(v => Number.isInteger(v) && v > 0)) throw new Error('LayerNorm normalized_shape 必须是正整数或正整数数组');
        if (normalized.length > output.length - 1 || output.slice(-normalized.length).join(',') !== normalized.join(',')) throw new Error('LayerNorm normalized_shape 必须匹配输入尾部维度');
        if (Number(n.params.elementwise_affine ?? 1)) count = product(normalized) * 2;
      } else if (n.op === 'GroupNorm') {
        if (output.length < 3) throw new Error('GroupNorm 需要 NCHW 类输入'); const groups = integer('num_groups', 1, 1, output[1]); if (output[1] % groups) throw new Error('num_channels 必须能被 num_groups 整除'); if (Number(n.params.affine ?? 1)) count = output[1] * 2;
      } else if (n.op.startsWith('AdaptiveAvgPool') || n.op.startsWith('AdaptiveMaxPool')) {
        const rank = Number(n.op.match(/\d$/)?.[0] || 2) + 2; if (output.length !== rank) throw new Error(`${n.op} 需要 ${rank} 维通道输入`);
        const raw = n.params.output_size ?? 1, sizes: number[] = (Array.isArray(raw) ? raw : Array(rank - 2).fill(raw)).map(Number);
        if (sizes.length !== rank - 2 || !sizes.every(v => Number.isInteger(v) && v > 0 && v <= 256)) throw new Error('自适应池化 output_size 无效'); output = [output[0], output[1], ...sizes];
      } else if (n.op === 'Flatten') output = [output[0], product(output.slice(1))];
      else if (n.op === 'Unsqueeze') { const dim = integer('dim', 0, -output.length - 1, output.length); const index = dim < 0 ? dim + output.length + 1 : dim; output = [...output.slice(0, index), 1, ...output.slice(index)]; }
      else if (n.op === 'Squeeze') {
        const raw = n.params.dim ?? 'all';
        if (raw === 'all') output = output.filter(size => size !== 1);
        else {
          const dims = Array.isArray(raw) ? raw : [raw], rank = Math.max(output.length, 1);
          if (!dims.every(dim => typeof dim === 'number' && Number.isInteger(dim) && dim >= -rank && dim < rank)) throw new Error(`squeeze dim 需要在 [${-rank}, ${rank - 1}] 范围内的整数`);
          const indices = (dims as number[]).map(dim => (dim + rank) % rank);
          if (new Set(indices).size !== indices.length) throw new Error('squeeze dim 不能重复指定同一维度');
          output = output.filter((size, index) => !indices.includes(index) || size !== 1);
        }
      }
      else if (n.op === 'Slice' || n.op === 'Select') {
        const dim = (integer('dim', 1, -output.length, output.length - 1) + output.length) % output.length;
        if (dim === 0) throw new Error('Slice / Select 不能改变 batch 维度');
        if (n.op === 'Select') { integer('index', -1, -output[dim], output[dim] - 1); output.splice(dim, 1); }
        else {
          const size = output[dim], step = integer('step', 1);
          const bound = (key: string, fallback: number) => { const raw = n.params[key] ?? 'none'; if (raw === 'none') return fallback; const value = integer(key, 0, -65536, 65536); return Math.min(size, Math.max(0, value < 0 ? size + value : value)); };
          const length = Math.max(0, Math.ceil((bound('end', size) - bound('start', 0)) / step));
          if (!length) throw new Error('Slice 输出为空'); output[dim] = length;
        }
      } else if (n.op === 'ConstantAdd') {
        const shape = n.params.shape, values = n.params.values;
        if (!output.length || !Array.isArray(shape) || shape.length > Math.min(5, output.length) || !shape.every(v => Number.isInteger(v) && v >= 1 && v <= 65536)) throw new Error('常量 shape 无效');
        if (!Array.isArray(values) || !values.length || values.length > 65536 || product(shape) !== values.length || !values.every(v => Number.isFinite(v) && Math.abs(v) <= 3.4028234663852886e38)) throw new Error('常量 values 必须是与 shape 匹配的有限 FP32 数组');
        constantElements += values.length; if (constantElements > 65536) throw new Error('模型常量缓冲区不能超过 65536 个元素');
        const offset = output.length - shape.length, aligned = [...Array(offset).fill(1), ...shape];
        if (aligned[0] !== 1) throw new Error('常量必须在 batch 维度广播');
        const sequenceDim = n.params.sequence_dim === undefined ? null : integer('sequence_dim', 1, 1, output.length - 1);
        if (sequenceDim !== null && sequenceDim < offset) throw new Error('sequence_dim 不在常量维度中');
        aligned.forEach((size, dim) => { if (dim === sequenceDim ? output[dim] > size : size !== 1 && size !== output[dim]) throw new Error(dim === sequenceDim ? '输入序列超过常量缓冲区容量' : '常量形状无法广播到输入'); });
      } else if (n.op === 'Linear') {
        if (output.length < 2) throw new Error('Linear 需要至少二维输入 [B,...,F]');
        const features = integer('out_features', 10); count = (output.at(-1)! + 1) * features; output = [...output.slice(0, -1), features];
      } else if (n.op === 'Bilinear') {
        if (input.length !== 2 || input.some(s => s.length !== 2)) throw new Error('Bilinear 需要两个二维输入'); const features = integer('out_features', 10), in2 = integer('in2_features', input[1][1]); if (input[1][1] !== in2 || input[0][0] !== input[1][0]) throw new Error('Bilinear 输入 batch 与 in2_features 不匹配'); count = input[0][1] * in2 * features + features; output = [output[0], features];
      } else if (n.op === 'Embedding') {
        if (output.length < 2) throw new Error('Embedding 需要索引序列输入');
        if (graph.nodes.find(node => node.id === parents[0])?.op !== 'Input') throw new Error('Embedding 输入必须直接来自 Input 层');
        const features = integer('embedding_dim', 32, 1, 4096), entries = integer('num_embeddings', 100, 1, 1e6); count = entries * features; output = [...output, features];
      } else if (n.op === 'Upsample') {
        if (output.length < 3) throw new Error('Upsample 需要通道和空间维度'); const raw = n.params.scale_factor ?? 2, scales: number[] = (Array.isArray(raw) ? raw : Array(output.length - 2).fill(raw)).map(Number); if (scales.length !== output.length - 2 || !scales.every(v => Number.isFinite(v) && v > 0)) throw new Error('scale_factor 无效'); output = [output[0], output[1], ...output.slice(2).map((v, i) => Math.max(1, Math.floor(v * scales[i])))];
      } else if (n.op === 'Transformer' || n.op === 'MultiHeadAttention') {
        if (output.length !== 3) throw new Error('注意力需要三维输入 [B,S,E]');
        const embed = integer('embed_dim', 64, 1, 4096), heads = integer('num_heads', 1, 1, 16);
        const { type } = attentionConfig(n.params);
        if (!Object.hasOwn(ATTENTION_TYPES, type)) throw new Error('attention_type 无效');
        const kvHeads = integer('kv_heads', type === 'multi_query' ? 1 : heads, 1, 16), branches = integer('branches', 1, 1, 8);
        if (kvHeads > heads || heads % kvHeads !== 0) throw new Error('kv_heads 不能大于 num_heads，且 num_heads 必须能被 kv_heads 整除');
        if (['self', 'multi_branch'].includes(type) && kvHeads !== heads) throw new Error('标准自注意力需要 kv_heads = num_heads；共享 KV 请使用 MQA / GQA');
        if (type === 'multi_query' && kvHeads !== 1) throw new Error('Multi-Query Attention 需要 kv_heads = 1');
        if (type === 'multi_branch' && branches < 2) throw new Error('Multi-Branch Attention 需要 branches ≥ 2');
        if (output[2] !== embed) throw new Error(`输入特征维度 ${output[2]} 与 embed_dim ${embed} 不一致`);
        if (embed % heads !== 0) throw new Error('embed_dim 必须能被 num_heads 整除');
        if (type === 'cross' && (input[1].length !== 3 || input[1][0] !== output[0] || input[1][2] !== embed)) throw new Error('Context 需要 [B,S_kv,E]，batch 和 embed_dim 与 Query 一致；序列长度可不同');
        const overrides = allEdges.filter(e => isProjectionPort(e.targetPort));
        if (new Set(overrides.map(e => e.targetPort)).size !== overrides.length) throw new Error('每个 Q/K/V 端口只能指定一个输入对象');
        const portShapes = Object.fromEntries(projectionPorts(n).map(port => {
          const edge = overrides.find(e => e.targetPort === port.id), shape = edge ? edgeOutputShape(graph, edge, layers)! : input[type === 'cross' && port.role !== 'Q' ? 1 : 0];
          if (shape.length !== 3 || shape[0] !== output[0] || shape[2] !== embed || (port.role === 'Q' && shape[1] !== output[1])) throw new Error(`${port.label} 输入需要 [B,S,embed_dim]，Q 序列长度与 Query 一致`);
          return [port.id, shape];
        }));
        for (let b = 0; b < branches; b++) {
          const lengths = projectionPorts(n).filter(p => p.branch === b && p.role !== 'Q').map(p => portShapes[p.id][1]);
          if (new Set(lengths).size !== 1) throw new Error('同一分支 K/V 的序列长度必须一致');
        }
        const dropout = n.params.dropout ?? 0.1;
        if (typeof dropout !== 'number' || !Number.isFinite(dropout) || dropout < 0 || dropout >= 1) throw new Error('dropout 需要在 [0,1) 范围内');
        const attentionElements = output[0] * heads * output[1] * Array.from({ length: branches }, (_, b) => portShapes[`b${b}:k0`][1]).reduce((a, b) => a + b, 0);
        if (attentionElements > 16e6) throw new Error('注意力矩阵超过 1600 万元素，请减少序列长度或头数');
        activationBytes += attentionElements * 4;
        const kvDim = kvHeads * (embed / heads);
        count = branches * (2 * embed * embed + 2 * embed * kvDim + 2 * embed + 2 * kvDim);
        if (n.op === 'Transformer') {
          integer('norm_first', 1, 0, 1);
          if (!['relu', 'gelu'].includes(String(n.params.activation ?? 'gelu'))) throw new Error('编码器 activation 需要为 relu 或 gelu');
          const ff = integer('ff_dim', 128, 1, 16384);
          if (product(output.slice(0, 2)) * ff > 16e6) throw new Error('FFN 中间激活超过 1600 万元素');
          activationBytes += product(output.slice(0, 2)) * ff * 4;
          count += 2 * embed * ff + ff + 5 * embed;
        }
      } else if (['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout'].includes(n.op)) {
        const p = n.params.p ?? 0.3; if (typeof p !== 'number' || p < 0 || p >= 1) throw new Error('Dropout p 需要在 [0,1) 范围内');
      } else if (['PReLU'].includes(n.op)) {
        const numParameters = integer('num_parameters', 1, 1, 4096), init = n.params.init ?? 0.25;
        if (typeof init !== 'number' || !Number.isFinite(init)) throw new Error('PReLU init 无效'); count = numParameters;
      } else if (['Softmax', 'LogSoftmax'].includes(n.op)) {
        const dim = Number(n.params.dim ?? -1); if (!Number.isInteger(dim) || dim < -output.length || dim >= output.length) throw new Error('dim 超出输入维度');
      } else if (n.op === 'Add') {
        if (input.length < 2 || input.some(s => s.join(',') !== output.join(','))) throw new Error('Add 至少需要两个完全相同形状的输入');
      } else if (n.op === 'Concat') {
        if (input.length < 2) throw new Error('Concat 至少需要两个输入'); const dim = integer('dim', 1, 1, output.length - 1);
        if (input.some(s => s.length !== output.length || s.some((v, i) => i !== dim && v !== output[i]))) throw new Error('Concat 非拼接维度必须一致');
        output[dim] = input.reduce((sum, s) => sum + s[dim], 0);
      }
      if (product(output) > 16e6 || count > 50e6) throw new Error('该层超过本地工作台限制（1600 万激活 / 5000 万参数）');
      layers[id] = { input, output, parameters: count }; parameters += count; activationBytes += product(output) * 4;
      if (n.op === 'Linear' && n.id !== graph.edges.find(e => e.target === outputNode?.id)?.source && input[0].at(-1)! >= 128 && output.at(-1)! < input[0].at(-1)! * 0.1) add('BOTTLENECK', '隐藏层特征维度骤降超过 90%，可能丢失信息', id, 'warning');
      depth[id] = ['Add', 'Transformer'].includes(n.op) ? 0 : Math.max(0, ...parents.map(p => depth[p] || 0)) + (['Conv2d', 'Linear'].includes(n.op) ? 1 : 0);
      if (depth[id] === 8) add('DEEP_NO_SKIP', '连续 8 个参数层缺少残差路径，存在梯度衰减风险', id, 'warning');
      if (outputNode && !ancestors.has(id)) add(n.op === 'Input' ? 'INPUT_UNUSED' : 'UNUSED', '此层未连接到模型输出', id, n.op === 'Input' ? 'error' : 'warning');
      if (n.op === 'Output' && graph.edges.some(e => e.source === id)) throw new Error('输出层不能连接下游');
    } catch (e) { add('SHAPE', (e as Error).message, id); }
  }
  const batches = graph.nodes.filter(n => n.op === 'Input' && layers[n.id]).map(n => layers[n.id].output[0]);
  if (new Set(batches).size > 1) add('INPUT_BATCH', '所有模型输入的 batch 维度必须一致');
  if (parameters > 50e6) add('PARAMETER_LIMIT', '模型总参数超过 5000 万，请减少头维度或分支数');
  if (parameters > 0) Object.entries(layers).forEach(([id, info]) => { if (info.parameters > 1e4 && info.parameters / parameters > 0.7) add('PARAMETER_HOG', `该层占总参数 ${(info.parameters / parameters * 100).toFixed(0)}%，考虑池化或降维`, id, 'warning'); });
  return { layers, order, diagnostics, parameters, activationBytes, valid: !diagnostics.some(d => d.level === 'error') };
}

export function diagnoseMetrics(metrics: Metric[], patience = 5): Diagnostic[] {
  const alerts: Diagnostic[] = []; if (!metrics.length) return alerts;
  const add = (code: string, message: string, level: Diagnostic['level'] = 'warning', nodeId?: string) => alerts.push({ id: code + (nodeId || ''), code, message, level, nodeId });
  const last = metrics.at(-1)!;
  if (![last.trainLoss, last.valLoss, last.gradNorm].every(Number.isFinite)) add('NON_FINITE', '损失或梯度出现 NaN / Inf，请停止训练并降低学习率', 'error');
  const window = metrics.slice(-patience);
  if (window.length >= patience) {
    const first = window[0];
    const gap = (last.valLoss - last.trainLoss) / Math.max(Math.abs(last.trainLoss), 0.05);
    if (gap > 0.3 && last.trainLoss < first.trainLoss && last.valLoss > first.valLoss * 1.02) add('OVERFIT', '验证损失上升、训练损失下降，泛化差距持续扩大。建议早停或增加正则化');
    const reduction = (first.trainLoss - last.trainLoss) / Math.max(Math.abs(first.trainLoss), 1e-8);
    if (last.trainLoss > 0.1 && reduction < 0.01) add('NOT_CONVERGING', '最近窗口训练损失改善不足 1%，可能处于平台期或未收敛。检查学习率与数据');
    if (last.trainLoss > first.trainLoss * 1.5) add('DIVERGING', '训练损失快速上升，可能发散。建议降低学习率', 'error');
  }
  if (last.gradNorm > 100) add('GRADIENT_EXPLOSION', '全局梯度范数超过 100，建议使用梯度裁剪', 'error');
  else if (last.gradNorm < 1e-7 && last.trainLoss > 0.1) add('GRADIENT_VANISHING', '梯度范数极低，可能存在梯度消失');
  Object.entries(last.layerGradients || {}).forEach(([id, norm]) => { if (!Number.isFinite(norm) || norm > 100) add('LAYER_GRADIENT', `层 ${id} 梯度异常：${norm.toPrecision(3)}`, 'error', id); });
  Object.entries(last.deadRelu || {}).forEach(([id, ratio]) => { if (ratio > 0.9) add('DEAD_RELU', `层 ${id} 零激活比例 ${(ratio * 100).toFixed(0)}%，建议尝试 GELU`, 'warning', id); });
  return alerts;
}
