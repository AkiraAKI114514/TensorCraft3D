import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import AttentionMatrix, { matrixStep } from './AttentionMatrix';
import AttentionInspector from './AttentionInspector';
import { I18nProvider } from './i18n';
import type { AttentionMatrix as Matrix, AttentionSnapshot, HeadSummary } from './inferenceTypes';

const matrix: Matrix = {
  nodeId: 'attention', shape: [2, 3], dtype: 'float32', elements: 6, finiteCount: 5, nonFiniteCount: 1,
  stats: { min: -2, max: 2, mean: 0, std: 1 }, histogram: { edges: [-2, 0, 2], counts: [2, 3] },
  rowStart: 0, columnStart: 1, slice: { indices: [], shape: [2, 3], rows: 2, columns: 2, values: [[-2, 0], [2, null]], truncated: true }
};
const snapshot: AttentionSnapshot = {
  nodeId: 'attention', branch: 1, head: 3, kvHead: 1, numHeads: 4, kvHeads: 2, branches: 2,
  headDim: 2, queryLength: 2, keyLength: 3, scale: 1 / Math.sqrt(2), mask: 'none', dropout: 0,
  queryStart: 0, keyStart: 1, scoreSource: 'projected-qk', outputSource: 'scaled_dot_product_attention',
  tensors: { q: matrix, k: matrix, v: matrix, scores: matrix, probabilities: matrix, headOutput: matrix, branchOutput: matrix, mergedOutput: matrix }
};
const overview: HeadSummary[] = [0, 1, 2, 3].map(head => ({
  head, kvHead: head >> 1, rows: 2, columns: 2, values: head === 0 ? [[1, 0], [null, 0.5]] : [[0.5, 0.5], [0.2, 0.8]],
  entropy: head === 0 ? null : 0.25, peak: head === 0 ? null : 0.9,
}));

describe('bounded attention presentation', () => {
  it('encodes probability with a fixed scale and signed matrices symmetrically', () => {
    expect(matrixStep(0, 1, true)).toBe(0);
    expect(matrixStep(1, 1, true)).toBe(6);
    expect(matrixStep(0.5, 1, true)).toBe(3);
    expect(matrixStep(-2, 2, false)).toBe(matrixStep(2, 2, false));
    expect(matrixStep(null, 2, false)).toBeNull();
    expect(matrixStep(Infinity, 2, false)).toBeNull();
    expect(matrixStep(0, 0, false)).toBe(0);
  });

  it('labels bounded window offsets and makes every value keyboard/table readable', () => {
    const markup = renderToStaticMarkup(<AttentionMatrix matrix={matrix} title="Scaled scores" columns="Key" />);
    expect(markup).toContain('tabindex="0"');
    expect(markup).toContain('Query 0 · Key 1：-2');
    expect(markup).toContain('有界窗口，非完整矩阵');
    expect(markup).toContain('矩阵数值表 · 当前窗口');
    expect(markup).toContain('纹理辅助');
    expect(markup).toContain('matrix-neutral');
    expect(markup).toContain('matrix-negative-6');
    expect(markup).toContain('matrix-nonfinite');
    expect(markup).not.toContain('NaN');
  });

  it('gives each matrix independent texture IDs and an explicit 0..1 probability legend', () => {
    const markup = renderToStaticMarkup(<><AttentionMatrix matrix={matrix} title="A" probability /><AttentionMatrix matrix={matrix} title="B" probability /></>);
    const patternIds = Array.from(markup.matchAll(/<pattern[^>]+id="([^"]+)"/g), match => match[1]);
    expect(new Set(patternIds).size).toBe(patternIds.length);
    expect(markup).toContain('概率色阶：固定 0 到 1');
  });

  it('separates KV grouping, recomputed probabilities and actual Transformer output', () => {
    const markup = renderToStaticMarkup(<AttentionInspector snapshot={snapshot} transformer />);
    expect(markup).toContain('B2 / H4');
    expect(markup).toContain('使用 KV 组 2 / 2');
    expect(markup).toContain('无 mask · dropout=0');
    expect(markup).toContain('可能与融合 SDPA 有浮点差异');
    expect(markup).toContain('最终输出还包含残差与 FFN');
    expect(markup).toContain('分支输出 · Wₒ 后');
    expect(markup).toContain('Head 输出 · 实际 SDPA');
    // 没有掩码时不出现任何掩码标记。
    expect(markup).not.toContain('matrix-mask-layer');
    expect(markup).not.toContain('causal');
  });

  it('marks the keys a causal mask hides by their position in the full matrix, not in the window', () => {
    // 窗口从 Key 1 开始：Query 0 的两个格子（Key 1、2）都在自己之后，Query 1 只有 Key 2 在之后。
    const markup = renderToStaticMarkup(<AttentionInspector snapshot={{ ...snapshot, mask: 'causal' }} transformer={false} />);
    expect(markup).toContain('causal 掩码 · dropout=0');
    expect(markup).not.toContain('无 mask');
    expect(markup).toContain('斜线格被 causal 掩码屏蔽');
    expect(markup.match(/class="matrix-mask-layer"/g)).toHaveLength(3);
    expect(markup).toContain('Query 0 · Key 1：-2 · 已被 causal 掩码屏蔽');
    expect(markup).toContain('aria-label="Softmax 概率 · Query 1 · Key 1：2"');
    expect(markup.match(/class="matrix-masked-value"/g)).toHaveLength(3);
  });

  it('only masks score and probability matrices, and translates the mask wording', () => {
    const projections = renderToStaticMarkup(<AttentionMatrix matrix={matrix} title="Q" />);
    expect(projections).not.toContain('matrix-mask-layer');
    const english = renderToStaticMarkup(<I18nProvider initialLanguage="en"><AttentionInspector snapshot={{ ...snapshot, mask: 'causal', overview }} transformer={false} onSelectHead={() => undefined} /></I18nProvider>);
    expect(english).not.toMatch(/[㐀-鿿]/);
    expect(english).toContain('masked by the causal mask');
    expect(english).toContain('All heads · B2 · probability window');
  });

  it('sets every head of the branch side by side and marks the one the detail shows', () => {
    const live = renderToStaticMarkup(<AttentionInspector snapshot={{ ...snapshot, overview }} transformer={false} onSelectHead={() => undefined} />);
    expect(live).toContain('全部 Head · B2 · 概率窗口');
    expect(live.match(/aria-pressed="(true|false)"/g)).toEqual(['aria-pressed="false"', 'aria-pressed="false"', 'aria-pressed="false"', 'aria-pressed="true"']);
    expect(live).toContain('aria-label="H4：平均熵 0.25，平均最大概率 0.90"');
    expect(live).toContain('aria-label="H1：平均熵 —，平均最大概率 —"');
    expect(live).toContain('需要重新运行');
    expect(live).not.toContain('aria-disabled');
    // The fixed 0–1 scale: one is the darkest step, zero is neutral, a missing value is marked.
    expect(live).toContain('matrix-positive-6');
    expect(live).toContain('matrix-neutral');
    expect(live).toContain('matrix-nonfinite');
    // A snapshot only holds the captured head's detail: the tiles stay readable but choose nothing.
    const offline = renderToStaticMarkup(<AttentionInspector snapshot={{ ...snapshot, overview }} transformer={false} />);
    expect(offline.match(/aria-disabled="true"/g)).toHaveLength(4);
    expect(offline).toContain('快照只含采集时那个 head 的详情');
    // One head has nothing to compare, and an older report has no overview at all.
    expect(renderToStaticMarkup(<AttentionInspector snapshot={{ ...snapshot, overview: overview.slice(3) }} transformer={false} />)).not.toContain('head-overview');
    expect(renderToStaticMarkup(<AttentionInspector snapshot={snapshot} transformer={false} />)).not.toContain('head-overview');
  });
});
