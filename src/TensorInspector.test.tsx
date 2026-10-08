import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import TensorHistogram from './TensorHistogram';
import TensorInspector from './TensorInspector';
import { tensorNumber } from './inferenceTypes';
import { PRESETS } from './presets';
import type { TrainedModelMetadata } from './trainedModel';

const trainedModel: TrainedModelMetadata = {
  modelId: 'model-fixture', trainingRunId: 'training-fixture', graphFingerprint: 'fingerprint', createdAt: '2026-10-08T00:00:00Z',
  device: 'CPU', dataset: 'csv', seed: 42, epochsCompleted: 4, weightsEpoch: 1, reason: 'early_stopping', preprocessing: 'csv-standardized', storage: 'backend-memory'
};

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
    expect(markup).toContain('不加载导入源码的原始权重');
    expect(markup).toContain('运行单样本推理');
  });

  it('shows retained model identity, best epoch and temporary storage without claiming imported weights', () => {
    const markup = renderToStaticMarkup(<TensorInspector graph={PRESETS.mlp()} trainedModel={trainedModel} trainedReady selected="layer_1" active ready cuda={false} valid blocked={false} onBusy={() => {}} />);
    expect(markup).toContain('value="trained" selected=""');
    expect(markup).toContain('Model model-fixture');
    expect(markup).toContain('Training training-fixture');
    expect(markup).toContain('早停最佳权重');
    expect(markup).toContain('权重轮次 1');
    expect(markup).toContain('不加载导入源码的原始权重');
    expect(markup).toContain('64 MiB');
    expect(markup).not.toContain('当前计算图没有已确认的训练权重');
  });

  it('does not enable trained inference on a backend without the capability', () => {
    const markup = renderToStaticMarkup(<TensorInspector graph={PRESETS.mlp()} trainedModel={trainedModel} selected="layer_1" active ready cuda={false} valid blocked={false} onBusy={() => {}} />);
    expect(markup).toContain('后端暂不支持训练权重推理');
    expect(markup).toContain('class="button primary wide" disabled=""');
  });
});
