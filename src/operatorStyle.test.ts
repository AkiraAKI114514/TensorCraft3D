import { describe, expect, it } from 'vitest';
import { analyze } from './analysis';
import { PRESETS } from './presets';
import { COLORS, OPS, type Graph } from './types';
import { FAMILIES, familyOf, layoutTag, legendRows } from './operatorStyle';

const rowsOf = (name: string, options?: { showKernels?: boolean }) => {
  const graph = PRESETS[name]();
  return legendRows(graph, analyze(graph), options);
};
const names = (name: string) => rowsOf(name).map(row => row.names.join(' · '));

describe('operator presentation', () => {
  it('gives every operator one family, and every family one colour', () => {
    for (const op of OPS) expect(familyOf(op).ops).toContain(op);
    for (const family of FAMILIES) expect(new Set(family.ops.map(op => COLORS[op])).size, family.id).toBe(1);
    expect(new Set(FAMILIES.flatMap(family => family.ops)).size).toBe(OPS.length);
  });

  it('builds the DeepSeek-V3-style legend from its operators, in the order data reaches them', () => {
    expect(names('deepseek')).toEqual(['输入 / 输出', '词嵌入', '归一化与正则化', 'MLA 多头潜在注意力', '残差连接', '全连接', '激活', '逐元素门控', 'MoE 混合专家', '形状变换', '数据流']);
    const rows = rowsOf('deepseek'), row = (key: string) => rows.find(item => item.key === key)!;
    expect(row('latent')).toMatchObject({ glyph: 'waist', color: COLORS.LatentAttention, note: { text: 'KV 低秩压缩 · 解耦 RoPE', literal: false } });
    expect(row('moe')).toMatchObject({ glyph: 'experts', color: COLORS.MoE });
    // Families without a fixed explanation list the operators the model actually uses.
    expect(row('norm').note).toEqual({ text: 'RMSNorm', literal: true });
    expect(row('activation').note).toEqual({ text: 'SiLU', literal: true });
    expect(row('input')).toMatchObject({ glyph: 'io', color: COLORS.Input, secondary: COLORS.Output });
  });

  it('shows only what a convolutional model draws', () => {
    expect(names('cnn')).toEqual(['输入 / 输出', '卷积', '归一化与正则化', '激活', '池化与重采样', '形状变换', '全连接', '卷积窗口', '数据流']);
    expect(names('cnn')).not.toContain('MLA 多头潜在注意力');
    expect(rowsOf('cnn', { showKernels: false }).map(row => row.key)).not.toContain('kernel');
  });

  it('names the residual path when there is one, and an attention head\'s parts when there are heads', () => {
    expect(names('residual')).toContain('残差连接');
    expect(names('transformer')).toEqual(expect.arrayContaining(['Transformer 编码层', '注意力头内部']));
    // An Add of two independent branches is not a shortcut.
    const graph: Graph = { version: 1, name: 'sum', nodes: [
      { id: 'a', name: 'a', op: 'Input', params: { shape: [1, 8] }, position: { x: 0, y: 0 } },
      { id: 'b', name: 'b', op: 'Input', params: { shape: [1, 8] }, position: { x: 0, y: 0 } },
      { id: 'sum', name: 'sum', op: 'Add', params: {}, position: { x: 0, y: 0 } },
      { id: 'out', name: 'out', op: 'Output', params: {}, position: { x: 0, y: 0 } }
    ], edges: [{ id: 'e0', source: 'a', target: 'sum' }, { id: 'e1', source: 'b', target: 'sum' }, { id: 'e2', source: 'sum', target: 'out' }] };
    expect(legendRows(graph, analyze(graph)).find(row => row.key === 'residual')).toMatchObject({ names: ['逐元素相加'], glyph: 'merge' });
  });

  it.each(Object.keys(PRESETS))('never shows one colour twice in the legend of %s', name => {
    const colors = rowsOf(name).map(row => row.color);
    expect(new Set(colors).size).toBe(colors.length);
  });

  it('tags the layout from what the layers output', () => {
    const tag = (name: string) => layoutTag(analyze(PRESETS[name]()));
    expect([tag('cnn'), tag('transformer'), tag('deepseek'), tag('mlp')]).toEqual(['NCHW', 'BSE', 'BSE', 'NC']);
  });
});
