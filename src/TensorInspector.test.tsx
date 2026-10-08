import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import TensorHistogram from './TensorHistogram';
import TensorInspector from './TensorInspector';
import { tensorNumber } from './inferenceTypes';
import { PRESETS } from './presets';

describe('tensor observation presentation', () => {
  it('formats finite, very small and non-finite values explicitly', () => {
    expect(tensorNumber(0)).toBe('0');
    expect(tensorNumber(1.234567)).toBe('1.2346');
    expect(tensorNumber(1e-8)).toBe('1.000e-8');
    expect(tensorNumber(null)).toBe('非有限值');
    expect(tensorNumber(Infinity)).toBe('非有限值');
  });

  it('provides keyboard-readable histogram bins and a complete table', () => {
    const markup = renderToStaticMarkup(<TensorHistogram histogram={{ edges: [0, 1, 2], counts: [2, 5] }} />);
    expect(markup).toContain('tabindex="0"');
    expect(markup).toContain('1 – 2：5 个元素');
    expect(markup).toContain('分布数据表');
    expect(markup).toContain('纹理辅助');
    expect(markup).not.toContain('NaN');
  });

  it('identifies random weights and keeps inactive inspector mounted but hidden', () => {
    const markup = renderToStaticMarkup(<TensorInspector graph={PRESETS.mlp()} selected="layer_1" active={false} ready cuda={false} valid blocked={false} onBusy={() => {}} />);
    expect(markup).toContain('hidden=""');
    expect(markup).toContain('随机初始化权重');
    expect(markup).toContain('不使用此前训练或原始模型权重');
    expect(markup).toContain('运行单样本推理');
  });
});
