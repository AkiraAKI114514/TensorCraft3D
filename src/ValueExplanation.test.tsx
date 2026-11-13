import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import ValueExplanationView, { EXPLAINED_OPS, defaultExplainIndex, explainAxes } from './ValueExplanation';
import { I18nProvider } from './i18n';
import type { ValueExplanation } from './inferenceTypes';

const convolution: ValueExplanation = {
  nodeId: 'conv', op: 'Conv2d', index: [5, 0, 0], rule: 'weighted-sum', value: 0.75, recomputed: 0.7500000123, bias: 0.25, divisor: null,
  count: 18, padded: 10,
  terms: [
    { input: [2, 0, 1], x: 1, parameter: [5, 0, 1, 2], w: 0.5, term: 0.5 },
    { input: [3, 1, 1], x: -0.5, parameter: [5, 1, 2, 2], w: 0.5, term: -0.25 },
    { input: null, x: 0, parameter: [5, 0, 0, 0], w: 0.3, term: 0 },
  ],
  rest: { count: 15, sum: 0.25 },
  channels: [{ channel: 2, sum: 0.75 }, { channel: 3, sum: -0.25 }], channelRest: { count: 0, sum: 0 },
  inputNode: 'in', inputShape: [4, 6, 6], instance: 0, instances: 1,
};
const pooling: ValueExplanation = {
  ...convolution, op: 'MaxPool2d', index: [1, 0, 0], rule: 'max', value: 31, recomputed: 31, bias: null, count: 4, padded: 5,
  terms: [{ input: [1, 1, 1], x: 31, parameter: null, w: null, term: 31 }, { input: [1, 1, 0], x: 30, parameter: null, w: null, term: 30 }],
  rest: { count: 2, sum: null }, channels: [], channelRest: null,
};

describe('explaining one output value', () => {
  it('starts in the middle of the map and names each output axis', () => {
    expect(defaultExplainIndex('Conv2d', [2, 16, 7, 8])).toEqual([0, 3, 3]);
    expect(defaultExplainIndex('Linear', [2, 5, 10])).toEqual([0, 0]);
    expect(explainAxes('Conv2d', 4)).toEqual(['输出通道', '输出行', '输出列']);
    expect(explainAxes('AvgPool1d', 3)).toEqual(['输出通道', '输出位置']);
    expect(explainAxes('Linear', 2)).toEqual(['输出特征']);
    expect(explainAxes('Linear', 3)).toEqual(['输出位置', '输出特征']);
    expect(EXPLAINED_OPS.has('Conv3d') && EXPLAINED_OPS.has('AdaptiveMaxPool2d') && !EXPLAINED_OPS.has('ConvTranspose2d') && !EXPLAINED_OPS.has('ReLU')).toBe(true);
  });

  it('writes a weighted sum as its largest terms, the rest, the bias and the float64 total', () => {
    const markup = renderToStaticMarkup(<ValueExplanationView explanation={convolution} layer="Conv" input="Image" />);
    expect(markup).toContain('Conv · 输出 [5, 0, 0] 的组成');
    expect(markup).toContain('y[5, 0, 0] = Σ 18 项 x·w + b');
    expect(markup).toContain('x 是 Image 在这次运行中的实测输出 [4, 6, 6]');
    expect(markup).toContain('其中 10 项落在补零区域');
    // A padded term has no input position; every weight names the element it came from.
    expect(markup).toContain('<th>补零</th>');
    expect(markup).toContain('title="W[5, 0, 1, 2]"');
    expect(markup).toContain('其余 15 项');
    expect(markup).toContain('偏置 b');
    // Bars are relative to the largest listed term, signed by colour.
    expect(markup).toContain('data-sign="positive" style="width:100%"');
    expect(markup).toContain('data-sign="negative" style="width:50%"');
    expect(markup).toContain('按输入通道汇总 · 2 个通道');
    expect(markup).not.toContain('折叠层');
  });

  it('marks the window value max pooling keeps and drops the weight column', () => {
    const markup = renderToStaticMarkup(<ValueExplanationView explanation={{ ...pooling, instance: 2, instances: 3, owner: 'first' }} layer="Pool" input="Conv" />);
    expect(markup).toContain('y[1, 0, 0] = max（4 个窗口值）');
    expect(markup).toContain('[1, 1, 1] · 最大');
    expect(markup).toContain('按 −∞ 处理，不参与比较');
    expect(markup).toContain('都不大于最大值');
    expect(markup).not.toContain('<th>w</th>');
    expect(markup).not.toContain('按输入通道汇总');
    expect(markup).toContain('这里拆的是最后一个（编号 2）');
    expect(markup).toContain('这一层共享 first 的权重');
  });

  it('divides average pooling by the window PyTorch counts, and translates', () => {
    const mean = { ...pooling, op: 'AvgPool2d', rule: 'mean' as const, divisor: 9, count: 9, recomputed: 1.1, value: 1.1, rest: { count: 7, sum: 0.2 } };
    expect(renderToStaticMarkup(<ValueExplanationView explanation={mean} layer="Pool" input="Conv" />)).toContain('y[1, 0, 0] = Σ 9 项 x ÷ 9');
    for (const explanation of [convolution, pooling, mean]) {
      const english = renderToStaticMarkup(<I18nProvider initialLanguage="en"><ValueExplanationView explanation={explanation} layer="Conv" input="Image" /></I18nProvider>);
      expect(english).not.toMatch(/[㐀-鿿]/);
    }
  });
});
