import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import TensorHistogram from './TensorHistogram';
import TensorInspector from './TensorInspector';
import { I18nProvider } from './i18n';
import { tensorNumber } from './inferenceTypes';
import { PRESETS } from './presets';
import type { TrainedModelMetadata } from './trainedModel';
import type { Graph, Layer } from './types';

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

  it('observes a layer inside a folded block at a chosen instance, and a block at its last instance\'s exit', () => {
    const layer = (id: string, op: Layer['op'], params: Layer['params'] = {}, extra: Partial<Layer> = {}): Layer => ({ id, name: id, op, params, position: { x: 0, y: 0 }, ...extra });
    const nodes = [layer('in', 'Input', { shape: [2, 8] }), layer('stack', 'Group', {}, { subgraph: 'cell', repeat: 3 }), layer('head', 'Linear', { out_features: 2 }), layer('out', 'Output')];
    const graph: Graph = {
      version: 1, name: 'Blocks', nodes, edges: nodes.slice(1).map((node, index) => ({ id: `e${index}`, source: nodes[index].id, target: node.id })),
      subgraphs: { cell: { id: 'cell', name: 'Cell', origin: 'manual', nodes: [layer('fc', 'Linear', { out_features: 8 }), layer('act', 'Tanh')], edges: [{ id: 'c0', source: 'fc', target: 'act' }] } },
    };
    const render = (drillPath: string[], selected: string, language: 'zh' | 'en' = 'zh') => renderToStaticMarkup(<I18nProvider initialLanguage={language}><TensorInspector graph={graph} drillPath={drillPath} selected={selected} active ready cuda={false} valid blocked={false} onBusy={() => {}} /></I18nProvider>);
    const inner = render(['cell'], 'fc');
    expect(inner).toContain('结构块实例 · Cell');
    expect(inner).toContain('aria-label="观测实例 · Cell" type="number" min="0" max="2" step="1" value="0"');
    expect(inner).toContain('观测展开后的层 stack/0/fc。');
    expect(inner).not.toContain('class="button primary wide" disabled=""');
    const block = render([], 'stack');
    expect(block).toContain('结构块的输出来自最后一个实例的出口层 stack/2/act。');
    expect(block).not.toContain('观测实例');
    expect(block).not.toContain('class="button primary wide" disabled=""');
    // A top-level layer needs no mapping, and a layer the level does not have cannot run.
    expect(render([], 'head')).not.toContain('tensor-block-scope');
    expect(render(['cell'], 'head')).toContain('class="button primary wide" disabled=""');
    const english = render(['cell'], 'fc', 'en');
    expect(english).toContain('Block instance · Cell');
    expect(english).toContain('aria-label="Observed instance · Cell"');
    expect(english).toContain('3 instances, numbered from 0');
    expect(english).toContain('Observing the expanded layer stack/0/fc.');
  });
});
