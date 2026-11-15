import { COLORS, OPS, type Analysis, type Graph, type Op } from './types';
import { isAttention } from './attentionConfig';
import { RESIDUAL_COLOR } from './flowGeometry';
import { residualEdges } from './graphRoutes';
import { WINDOW_OPS } from './receptiveField';

/**
 * How each operator is presented: the family it belongs to, the glyph that stands for its 3D shape,
 * and what the legend says about it. The scene, the legend and the layer library all draw an
 * operator in its family's colour (`COLORS`), so one table decides how an operator looks everywhere.
 */
export type Glyph = 'tensor' | 'slab' | 'membrane' | 'table' | 'waist' | 'experts' | 'heads' | 'residual' | 'merge' | 'shape' | 'io' | 'group' | 'flow' | 'kernel' | 'contract' | 'scale' | 'mask';
export interface OperatorFamily {
  id: string;
  /** Legend name (a zh-CN message key). */
  name: string;
  /** A fixed explanation (a message key); without one, the legend lists the family's operators present. */
  note?: string;
  glyph: Glyph;
  ops: readonly Op[];
}

export const FAMILIES: readonly OperatorFamily[] = [
  { id: 'input', name: '输入', glyph: 'io', ops: ['Input'] },
  { id: 'embedding', name: '词嵌入', note: '离散编号查表成向量', glyph: 'table', ops: ['Embedding'] },
  { id: 'constant', name: '固定常量', glyph: 'slab', ops: ['ConstantAdd', 'ConstantMul'] },
  { id: 'conv', name: '卷积', glyph: 'tensor', ops: ['Conv1d', 'Conv2d', 'Conv3d'] },
  { id: 'deconv', name: '转置卷积', glyph: 'tensor', ops: ['ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'] },
  { id: 'norm', name: '归一化与正则化', glyph: 'slab', ops: ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'RMSNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d', 'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout'] },
  { id: 'attention', name: '多头注意力', note: 'Q · K · V 投影，各头独立打分', glyph: 'heads', ops: ['MultiHeadAttention'] },
  { id: 'transformer', name: 'Transformer 编码层', note: '注意力 + 前馈，各带残差', glyph: 'heads', ops: ['Transformer'] },
  { id: 'latent', name: 'MLA 多头潜在注意力', note: 'KV 低秩压缩 · 解耦 RoPE', glyph: 'waist', ops: ['LatentAttention'] },
  { id: 'einsum', name: '张量收缩', glyph: 'contract', ops: ['Einsum'] },
  { id: 'scale', name: '缩放', glyph: 'scale', ops: ['Scale'] },
  { id: 'mask', name: '因果掩码', note: '遮住之后的位置', glyph: 'mask', ops: ['CausalMask'] },
  { id: 'moe', name: 'MoE 混合专家', note: '门控选 top-k 路由专家 + 共享专家', glyph: 'experts', ops: ['MoE'] },
  { id: 'linear', name: '全连接', glyph: 'tensor', ops: ['Linear', 'Bilinear'] },
  { id: 'activation', name: '激活', glyph: 'membrane', ops: ['ReLU', 'GELU', 'Sigmoid', 'Tanh', 'SiLU', 'LeakyReLU', 'ELU', 'SELU', 'Softplus', 'Softmax', 'LogSoftmax', 'PReLU', 'Hardsigmoid', 'Hardswish', 'Mish', 'Softsign'] },
  { id: 'gate', name: '逐元素门控', glyph: 'merge', ops: ['Multiply', 'Divide', 'GLU'] },
  { id: 'residual', name: '残差连接', note: '捷径与主路相加', glyph: 'residual', ops: ['Add'] },
  { id: 'concat', name: '拼接', glyph: 'merge', ops: ['Concat'] },
  { id: 'pool', name: '池化与重采样', glyph: 'tensor', ops: ['MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'Upsample'] },
  { id: 'reduce', name: '归约', glyph: 'slab', ops: ['Reduce'] },
  { id: 'shape', name: '形状变换', glyph: 'shape', ops: ['Flatten', 'Unflatten', 'Unsqueeze', 'Squeeze', 'Slice', 'Select', 'Identity'] },
  { id: 'group', name: '结构块', glyph: 'group', ops: ['Group'] },
  { id: 'output', name: '输出', glyph: 'io', ops: ['Output'] }
];

const familyByOp = new Map<Op, OperatorFamily>(FAMILIES.flatMap(family => family.ops.map(op => [op, family] as const)));
export const familyOf = (op: Op): OperatorFamily => familyByOp.get(op)!;
/** Every operator has a family; checked here so a new operator without one fails at load, not in a legend. */
for (const op of OPS) if (!familyByOp.has(op)) throw new Error(`Operator ${op} has no presentation family`);

/** The colours of an attention module's own parts, which its legend glyph shows. */
export const HEAD_COLORS = { q: '#319cac', k: '#d6a13d', v: '#a17cbb', merge: '#36a18a' } as const;
export const FLOW_COLOR = '#31adce';
export const KERNEL_COLOR = '#f97316';

export interface LegendRow {
  key: string;
  glyph: Glyph;
  color: string;
  /** A second colour the glyph shows: the output's, on the row shared by inputs and outputs. */
  secondary?: string;
  /** Message keys, joined with " · " when families share a colour. */
  names: string[];
  /** A message key, or the operator names present when `literal`. */
  note?: { text: string; literal: boolean };
}

/**
 * The legend for one level of a model: one row per family present, in the order data first reaches
 * it, then the data flow, the kernel window and the residual path when the scene draws them.
 * Families drawn in the same colour share a row, so no two rows show the same swatch.
 */
export function legendRows(graph: Graph, analysis: Pick<Analysis, 'order'>, options: { showKernels?: boolean } = {}): LegendRow[] {
  const order = [...analysis.order.filter(id => graph.nodes.some(node => node.id === id)), ...graph.nodes.map(node => node.id).filter(id => !analysis.order.includes(id))];
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  const present = new Map<string, { family: OperatorFamily; ops: Set<Op> }>();
  for (const id of order) {
    const op = byId.get(id)!.op, family = familyOf(op);
    if (!present.has(family.id)) present.set(family.id, { family, ops: new Set() });
    present.get(family.id)!.ops.add(op);
  }
  const rows: LegendRow[] = [];
  const byColor = new Map<string, LegendRow>();
  for (const { family, ops } of present.values()) {
    const color = COLORS[[...ops][0]];
    const note = family.note ? { text: family.note, literal: false } : family.glyph === 'io' ? undefined : { text: [...ops].join(' · '), literal: true };
    const shared = byColor.get(color);
    if (shared) {
      shared.names.push(family.name);
      shared.note = shared.note?.literal && note?.literal ? { text: `${shared.note.text} · ${note.text}`, literal: true } : shared.note ?? note;
      continue;
    }
    const row: LegendRow = { key: family.id, glyph: family.glyph, color, names: [family.name], note };
    // A residual Add stands for the whole shortcut; its row only says so when the model has one.
    if (family.id === 'residual' && !residualEdges(graph).size) { row.names = ['逐元素相加']; row.glyph = 'merge'; row.note = { text: [...ops].join(' · '), literal: true }; }
    byColor.set(color, row); rows.push(row);
  }
  // Inputs and outputs are two shades of one slate; they share a row, the glyph showing both.
  const input = rows.find(row => row.key === 'input'), output = rows.find(row => row.key === 'output');
  if (input && output) { input.names = ['输入 / 输出']; input.secondary = output.color; rows.splice(rows.indexOf(output), 1); }
  if (graph.nodes.some(node => isAttention(node.op))) rows.push({ key: 'heads', glyph: 'heads', color: HEAD_COLORS.q, names: ['注意力头内部'], note: { text: 'Q · K · V · 汇合', literal: false } });
  if (options.showKernels !== false && graph.nodes.some(node => WINDOW_OPS.has(node.op))) rows.push({ key: 'kernel', glyph: 'kernel', color: KERNEL_COLOR, names: ['卷积窗口'], note: { text: '一个输出元素读取的输入区域', literal: false } });
  if (graph.edges.length) rows.push({ key: 'flow', glyph: 'flow', color: FLOW_COLOR, names: ['数据流'], note: { text: '示意动画，不是实测值', literal: false } });
  return rows;
}

/** How a level's tensors are laid out, for the scene heading: from what the layers actually output. */
export function layoutTag(analysis: Pick<Analysis, 'layers'>): string {
  const ranks = new Set(Object.values(analysis.layers).map(layer => layer.output.length));
  return ranks.has(4) ? 'NCHW' : ranks.has(3) ? 'BSE' : 'NC';
}

export { RESIDUAL_COLOR };
