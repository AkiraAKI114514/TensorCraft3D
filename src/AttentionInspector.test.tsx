import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import AttentionMatrix, { matrixStep } from './AttentionMatrix';
import AttentionInspector from './AttentionInspector';
import type { AttentionMatrix as Matrix, AttentionSnapshot } from './inferenceTypes';

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
  });
});
