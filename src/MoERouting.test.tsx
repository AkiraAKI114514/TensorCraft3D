import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import MoERouting from './MoERouting';
import { I18nProvider } from './i18n';
import type { AttentionMatrix, RoutingSnapshot } from './inferenceTypes';

const window = (nodeId: string, values: number[][]): AttentionMatrix => ({
  nodeId, shape: [3, 4], dtype: 'float32', elements: 12, finiteCount: 12, nonFiniteCount: 0,
  stats: { min: 0, max: 1, mean: 0.25, std: 0.3 }, histogram: { edges: [0, 0.5, 1], counts: [8, 4] },
  rowStart: 0, columnStart: 0, slice: { indices: [], shape: [3, 4], rows: 3, columns: 4, values, truncated: false }
});
const routing: RoutingSnapshot = {
  nodeId: 'moe', positions: 3, experts: 4, topK: 2, sharedExperts: 1, scoreFunc: 'sigmoid', routeScale: 1, positionStart: 0, expertStart: 0,
  scores: window('scores', [[0.9, 0.2, 0.7, 0.1], [0.3, 0.8, 0.6, 0.2], [0.5, 0.4, 0.1, 0.9]]),
  weights: window('weights', [[0.56, 0, 0.44, 0], [0, 0.57, 0.43, 0], [0.36, 0, 0, 0.64]]),
  chosen: [[0, 2], [1, 2], [3, 0]], load: [2, 1, 2, 1]
};

describe('MoE routing view', () => {
  it('draws one load bar per expert with the mean, and names positions rather than tokens', () => {
    const markup = renderToStaticMarkup(<MoERouting routing={routing} />);
    expect(markup.match(/class="moe-load-bar"/g)).toHaveLength(4);
    expect(markup).toContain('class="moe-load-mean"');
    expect(markup).toContain('专家负载：2, 1, 2, 1');
    expect(markup).toContain('专家 0：2 个位置');
    expect(markup).toContain('位置 2：专家 3, 0');
    expect(markup).not.toMatch(/token/i);
  });

  it('labels the matrices by position and expert in English too', () => {
    const markup = renderToStaticMarkup(<I18nProvider initialLanguage="en"><MoERouting routing={routing} /></I18nProvider>);
    expect(markup).toContain('MoE routing · 2 of 4 experts per position');
    expect(markup).toContain('Routing weights · 0 for experts not chosen');
    expect(markup).toContain('Gate scores · sigmoid, before the bias');
    expect(markup).toContain('Position 0 · Expert 0');
  });
});
