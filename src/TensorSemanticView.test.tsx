import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import TensorSemanticView, { semanticKind } from './TensorSemanticView';
import { I18nProvider } from './i18n';
import type { ChannelPage, TensorSnapshot } from './inferenceTypes';

function snapshot(shape: number[], indices: number[], values: (number | null)[][], extra: Partial<TensorSnapshot> = {}): TensorSnapshot {
  const flat = values.flat().filter((value): value is number => value !== null);
  return {
    nodeId: 'layer', shape, dtype: 'float32', elements: shape.reduce((a, b) => a * b, 1), finiteCount: flat.length, nonFiniteCount: 0,
    stats: { min: Math.min(...flat), max: Math.max(...flat), mean: 0, std: 1 }, histogram: { edges: [0, 1], counts: [flat.length] },
    slice: { indices, shape: shape.slice(-2), rows: values.length, columns: values[0].length, values, truncated: false }, ...extra,
  };
}

const page: ChannelPage = {
  start: 16, count: 3, total: 19, selected: 17, shape: [2, 2], rows: 2, columns: 2, truncated: false,
  maps: [[[1, -1], [0, 2]], [[0.5, 0.5], [-2, null]], [[0, 0], [0, 0]]],
  stats: [{ min: -1, max: 2, mean: 0.5 }, { min: -2, max: 0.5, mean: -0.25 }, { min: 0, max: 0, mean: 0 }],
};
const render = (value: TensorSnapshot, english = false) => renderToStaticMarkup(english
  ? <I18nProvider initialLanguage="en"><TensorSemanticView snapshot={value} disabled={false} onSelectChannel={() => {}} /></I18nProvider>
  : <TensorSemanticView snapshot={value} disabled={false} onSelectChannel={() => {}} />);

describe('tensor semantic views', () => {
  it('reads a weight by its role: a filter\'s kernels across input channels, or a matrix', () => {
    const kernel = snapshot([20, 19, 2, 2], [2, 17], [[1, -1], [0, 2]], { channels: page });
    const markup = renderToStaticMarkup(<TensorSemanticView snapshot={kernel} subject="parameter" disabled={false} onSelectChannel={() => {}} />);
    expect(markup).toContain('卷积核 · 滤波器 2 · 输入通道 17');
    expect(markup).toContain('每格是这个滤波器在一个输入通道上的卷积核');
    expect(markup).not.toContain('特征图');
    const matrix = renderToStaticMarkup(<I18nProvider initialLanguage="en"><TensorSemanticView snapshot={snapshot([10, 64], [], [[1, 2]])} subject="parameter" disabled={false} onSelectChannel={() => {}} /></I18nProvider>);
    expect(matrix).toContain('Parameter matrix');
    expect(matrix).not.toContain('Feature vector');
  });

  it('reads the rank as feature map, volume, sequence or vector, and skips integer ids', () => {
    expect(semanticKind(snapshot([1, 8, 4, 4], [0, 0], [[1]]))).toBe('feature-map');
    expect(semanticKind(snapshot([1, 8, 2, 4, 4], [0, 0, 0], [[1]]))).toBe('volume');
    expect(semanticKind(snapshot([1, 6, 8], [0], [[1]]))).toBe('sequence');
    expect(semanticKind(snapshot([1, 8], [], [[1]]))).toBe('vector');
    // Embedding 的输入是类别 ID，按幅度上色没有意义。
    expect(semanticKind(snapshot([1, 6], [], [[3]], { dtype: 'int64' }))).toBeNull();
    expect(render(snapshot([1, 6], [], [[3]], { dtype: 'int64' }))).toBe('');
  });

  it('shows a feature map as a heatmap of the selected channel with a page of neighbouring channels', () => {
    const markup = render(snapshot([1, 19, 2, 2], [0, 17], [[0.5, 0.5], [-2, 0]], { channels: page }));
    expect(markup).toContain('语义视图');
    expect(markup).toContain('特征图 · 通道 17');
    // 行、列是高和宽，不是 Query / 特征。
    expect(markup).toContain('特征图 · 通道 17 · 行 1 · 列 0：-2');
    expect(markup).toContain('通道网格 · 通道 16–18 / 共 19');
    expect(markup.match(/aria-pressed="true"/g)).toHaveLength(1);
    expect(markup.match(/aria-pressed="false"/g)).toHaveLength(2);
    expect(markup).toContain('aria-label="通道 16：均值 0.5，范围 -1 – 2"');
    expect(markup).toContain('通道统计表 · 整个通道');
    // 非有限值不参与配色，也不会变成 NaN 文本。
    expect(markup).toContain('matrix-nonfinite');
    expect(markup).not.toContain('NaN');
  });

  it('puts every channel tile on the scale of the whole tensor so tiles are comparable', () => {
    // 整个张量的范围是 ±4：值为 2 的格子只能到一半的色阶，而不是按本通道的最大值拉满。
    const value = snapshot([1, 19, 2, 2], [0, 17], [[0.5, 0.5], [-2, 0]], { channels: page, stats: { min: -4, max: 4, mean: 0, std: 1 } });
    const markup = render(value);
    expect(markup).toContain('class="matrix-positive-3"');
    expect(markup).toContain('class="matrix-negative-3"');
    expect(markup).not.toContain('class="matrix-positive-6"');
    expect(markup).toContain('class="matrix-neutral"');
  });

  it('names sequence rows positions, never tokens, and has no channel grid', () => {
    const markup = render(snapshot([1, 2, 2], [0], [[1, 2], [3, 4]]));
    expect(markup).toContain('序列 · 位置 × 特征');
    expect(markup).toContain('序列 · 位置 × 特征 · 位置 1 · 特征 0：3');
    expect(markup).toContain('不带分词信息');
    expect(markup).not.toMatch(/token/i);
    expect(markup).not.toContain('通道网格');
  });

  it('shows a feature vector as one row and labels the depth of a volume', () => {
    expect(render(snapshot([1, 3], [], [[1, 2, 3]]))).toContain('特征向量 · 行 0 · 特征 2：3');
    const volume = render(snapshot([1, 4, 3, 2, 2], [0, 2, 1], [[1, 2], [3, 4]]));
    expect(volume).toContain('特征图 · 通道 2 · 深度 1');
    // 旧后端不返回通道页：只少一块网格，不报错。
    expect(volume).not.toContain('通道网格');
  });

  it('translates every title, axis name, readout and note', () => {
    for (const value of [snapshot([1, 19, 2, 2], [0, 17], [[0.5, 0.5], [-2, 0]], { channels: page }), snapshot([1, 4, 3, 2, 2], [0, 2, 1], [[1, 2], [3, 4]]), snapshot([1, 2, 2], [0], [[1, 2], [3, 4]]), snapshot([1, 3], [], [[1, 2, 3]])]) {
      const markup = render(value, true);
      expect(markup).not.toMatch(/[㐀-鿿]/);
      expect(markup).toContain('Semantic view');
    }
    expect(render(snapshot([1, 19, 2, 2], [0, 17], [[0.5, 0.5], [-2, 0]], { channels: page }), true)).toContain('Channel grid · channels 16–18 of 19');
  });
});
