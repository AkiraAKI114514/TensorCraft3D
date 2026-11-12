import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import LayerMetrics from './LayerMetrics';
import { I18nProvider } from './i18n';
import type { Metric } from './types';

const training = (epoch: number, layerGradients?: Record<string, number>, deadRelu?: Record<string, number>): Metric => ({ epoch, trainLoss: 1, valLoss: 1, accuracy: 0.5, gradNorm: 1, layerGradients, deadRelu, source: 'training' });
const run = [training(1, { dense: 0.8, 'block/0/n0': 0.2, 'block/1/n0': 0.6 }, { relu: 0.25 }), training(2, { dense: 0.4, 'block/0/n0': 0.1, 'block/1/n0': 0.3 }, { relu: 0.5 })];

describe('per-layer training metrics', () => {
  it('stays out of the way for layers that have nothing to measure', () => {
    expect(renderToStaticMarkup(<LayerMetrics metrics={run} keys={['flatten']} eligible={false} />)).toBe('');
    expect(renderToStaticMarkup(<LayerMetrics metrics={[]} keys={[]} eligible={false} />)).toBe('');
  });

  it('tells a parameter layer where its metrics will come from before any training', () => {
    const markup = renderToStaticMarkup(<LayerMetrics metrics={[]} keys={['dense']} eligible />);
    expect(markup).toContain('本层训练指标');
    expect(markup).toContain('真实训练后');
    expect(markup).not.toContain('role="slider"');
    // 演示数据没有逐层字段，不能因此显示成一条实测曲线。
    const demo: Metric[] = [{ ...training(1), source: 'demo' }];
    expect(renderToStaticMarkup(<LayerMetrics metrics={demo} keys={['dense']} eligible />)).not.toContain('实测');
  });

  it('plots the measured gradient norm with one tab stop, a readout and a full table', () => {
    const markup = renderToStaticMarkup(<LayerMetrics metrics={run} keys={['dense']} eligible />);
    expect(markup).toContain('实测');
    expect(markup).toContain('梯度范数 · 每轮批次均值，裁剪前');
    expect(markup.match(/tabindex="0"/g)).toHaveLength(1);
    expect(markup).toContain('aria-valuemin="1"');
    expect(markup).toContain('aria-valuenow="2"');
    expect(markup).toContain('aria-valuetext="第 2 轮：0.400"');
    expect(markup).toContain('<td>1</td><td>0.800</td>');
    expect(markup).toContain('<td>2</td><td>0.400</td>');
    // 这一层不是 ReLU，所以没有第二张图；实测指标也不需要「未在当前图上测量」的说明。
    expect(markup).not.toContain('ReLU 零激活比例');
    expect(markup).not.toContain('没有在当前计算图上测量');
    expect(markup).not.toContain('NaN');
  });

  it('shows the zero-activation ratio on its own fixed 0–100% scale', () => {
    const markup = renderToStaticMarkup(<LayerMetrics metrics={run} keys={['relu']} eligible />);
    expect(markup).toContain('ReLU 零激活比例 · 每轮批次均值');
    expect(markup).toContain('>100.0%<');
    expect(markup).toContain('aria-valuetext="第 2 轮：50.0%"');
    expect(markup).not.toContain('梯度范数');
  });

  it('summarises several expanded layers as a mean with a min–max band', () => {
    const markup = renderToStaticMarkup(<LayerMetrics metrics={run} keys={['block/0/n0', 'block/1/n0']} eligible />);
    expect(markup).toContain('2 个展开层 · 曲线为均值，色带为最小值到最大值');
    expect(markup).toContain('class="lm-band"');
    expect(markup).toContain('aria-valuetext="第 2 轮：均值 0.200，范围 0.100 – 0.300，2 个展开层"');
    expect(markup).toContain('<th>均值</th><th>最小</th><th>最大</th>');
  });

  it('marks imported metrics as not measured on the graph on screen', () => {
    const imported = run.map(metric => ({ ...metric, source: 'import' as const }));
    const markup = renderToStaticMarkup(<LayerMetrics metrics={imported} keys={['dense']} eligible />);
    expect(markup).toContain('导入指标');
    expect(markup).toContain('没有在当前计算图上测量');
    expect(markup).not.toContain('实测');
  });

  it('draws a single measured epoch as a marker rather than an empty plot', () => {
    const markup = renderToStaticMarkup(<LayerMetrics metrics={run.slice(0, 1)} keys={['dense']} eligible />);
    expect(markup).toContain('class="lm-marker"');
    expect(markup).not.toContain('<polyline');
    expect(markup).not.toContain('NaN');
  });

  it('translates every label, readout and table heading', () => {
    const imported = run.map(metric => ({ ...metric, source: 'import' as const }));
    for (const [metrics, keys] of [[run, ['dense']], [run, ['relu']], [imported, ['block/0/n0', 'block/1/n0']]] as [Metric[], string[]][]) {
      const markup = renderToStaticMarkup(<I18nProvider initialLanguage="en"><LayerMetrics metrics={metrics} keys={keys} eligible /></I18nProvider>);
      expect(markup).not.toMatch(/[㐀-鿿]/);
      expect(markup).toContain('Layer training metrics');
    }
    const empty = renderToStaticMarkup(<I18nProvider initialLanguage="en"><LayerMetrics metrics={[]} keys={['dense']} eligible /></I18nProvider>);
    expect(empty).not.toMatch(/[㐀-鿿]/);
  });
});
