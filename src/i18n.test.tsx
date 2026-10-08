import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { analyze, diagnoseMetrics } from './analysis';
import { PRESETS } from './presets';
import App from './App';
import CodeExport from './CodeExport';
import Parameters from './Parameters';
import PyTorchImport from './PyTorchImport';
import Connections from './Connections';
import AttentionMatrix from './AttentionMatrix';
import TensorHistogram from './TensorHistogram';
import type { AttentionMatrix as Matrix } from './inferenceTypes';
import { englishMessages, I18nProvider, LANGUAGE_STORAGE_KEY, loadLanguage, saveLanguage, translate } from './i18n';

vi.mock('./SceneView', () => ({ default: () => null }));
vi.mock('./TopologyView', () => ({ default: () => null }));
afterEach(() => vi.unstubAllGlobals());

describe('Chinese / English UI', () => {
  it('keeps Chinese as the standalone/default language', () => {
    const markup = renderToStaticMarkup(<App />);
    expect(markup).toContain('模型构建');
    expect(markup).toContain('aria-label="界面语言"');
    expect(loadLanguage()).toBe('zh');
    expect(translate('zh', '{count} 个结构错误', { count: 2 })).toBe('2 个结构错误');
  });

  it('translates the workbench and accessible labels without changing project content', () => {
    const markup = renderToStaticMarkup(<I18nProvider initialLanguage="en"><App /></I18nProvider>);
    expect(markup).toContain('Model builder');
    expect(markup).toContain('Export code');
    expect(markup).toContain('Training monitor');
    expect(markup).toContain('Waiting for training metrics');
    expect(markup).toContain('aria-label="Interface language"');
    expect(markup).toContain('aria-label="Undo"');
    expect(markup).not.toContain('模型构建');
    expect(markup).not.toContain('未开始');
    expect(markup).toContain(PRESETS.cnn().name);
  });

  it('translates nested controls but keeps exported JSON and parameter identifiers intact', () => {
    const graph = PRESETS.mlp();
    graph.name = '用户模型 / custom model';
    const markup = renderToStaticMarkup(<I18nProvider initialLanguage="en"><CodeExport graph={graph} valid tab="json" onTab={() => {}} onExport={() => {}} /><Parameters node={graph.nodes[0]} disabled={false} onUpdate={() => {}} /></I18nProvider>);
    expect(markup).toContain('Download .json');
    expect(markup).toContain(graph.name);
    expect(markup).toContain('Input shape');
    expect(markup).toContain('shape');
    expect(graph.name).toBe('用户模型 / custom model');
  });

  it('translates the import dialog and complete connection labels', () => {
    const graph = PRESETS.mlp();
    const markup = renderToStaticMarkup(<I18nProvider initialLanguage="en"><PyTorchImport online={false} disabled={false} onClose={() => {}} onImport={() => {}} /><Connections graph={graph} node={graph.nodes[1]} disabled={false} onConnect={() => {}} onRemove={() => {}} onRole={() => {}} /></I18nProvider>);
    expect(markup).toContain('Import code · automatic modeling');
    expect(markup).toContain('aria-label="PyTorch source code"');
    expect(markup).toContain('Supported scope and import method');
    expect(markup).toContain('aria-label="Select input object"');
    expect(markup).toContain('aria-label="Add output connection"');
    expect(markup).not.toContain('SelectInputobject');
    expect(markup).not.toContain('导入代码');
  });

  it('translates matrix non-finite values and histogram accessible readouts', () => {
    const matrix: Matrix = { nodeId: 'test', shape: [1, 1], dtype: 'float32', elements: 1, finiteCount: 0, nonFiniteCount: 1, stats: { min: null, max: null, mean: null, std: null }, histogram: { edges: [0, 1], counts: [0] }, rowStart: 0, columnStart: 0, slice: { indices: [], shape: [1, 1], rows: 1, columns: 1, values: [[null]], truncated: false } };
    const markup = renderToStaticMarkup(<I18nProvider initialLanguage="en"><AttentionMatrix matrix={matrix} title="Matrix" /><TensorHistogram histogram={{ edges: [0, 1, 2], counts: [2, 5] }} /></I18nProvider>);
    expect(markup.toLowerCase()).toContain('non-finite');
    expect(markup).toContain('Query 0 · Features 0: Non-finite value');
    expect(markup).toContain('1 – 2: 5 elements');
    expect(markup).not.toContain('非有限值');
    expect(markup).not.toContain('特征');
  });

  it('translates already-stored statuses and diagnostics, including nested messages', () => {
    expect(translate('en', '训练中 · cuda:0')).toBe('Training · cuda:0');
    expect(translate('en', 'PyTorch 已就绪 · CUDA 可用')).toBe('PyTorch ready · CUDA available');
    expect(translate('en', '3 个结构错误')).toBe('3 structure errors');
    expect(translate('en', 'BatchNorm1d 需要 2 或 3 维通道输入')).toBe('BatchNorm1d requires a 2- or 3-dimensional channel input');
    expect(translate('en', '结构错误：embed_dim 必须能被 num_heads 整除')).not.toMatch(/[㐀-鿿]/);
    const graph = PRESETS.mlp();
    graph.nodes[1].params.out_features = -1;
    for (const diagnostic of analyze(graph).diagnostics) expect(translate('en', diagnostic.message)).not.toMatch(/[㐀-鿿]/);
    const metrics = Array.from({ length: 5 }, (_, i) => ({ epoch: i + 1, trainLoss: 1 - i * 0.1, valLoss: 1 + i * 0.2, gradNorm: 200, accuracy: 0.2, source: 'demo' as const }));
    for (const diagnostic of diagnoseMetrics(metrics)) expect(translate('en', diagnostic.message)).not.toMatch(/[㐀-鿿]/);
  });

  it('preserves unknown server errors and interpolation data', () => {
    expect(translate('en', '服务返回的未知错误')).toBe('服务返回的未知错误');
    expect(translate('en', 'custom/model: 未知输入')).toBe('custom/model: 未知输入');
    expect(translate('en', '训练中 · {device}', { device: '用户自定义设备' })).toBe('Training · 用户自定义设备');
    expect(translate('en', '__proto__')).toBe('__proto__');
    expect(Object.keys(englishMessages).length).toBeGreaterThan(300);
    for (const [source, english] of Object.entries(englishMessages)) {
      expect(english.trim(), source).not.toBe('');
      expect([...english.matchAll(/\{(\w+)\}/g)].map(m => m[1]).sort(), source).toEqual([...source.matchAll(/\{(\w+)\}/g)].map(m => m[1]).sort());
    }
  });

  it('uses a separate preference key and handles invalid or unavailable storage', () => {
    const getItem = vi.fn(() => 'en'), setItem = vi.fn();
    vi.stubGlobal('window', { localStorage: { getItem, setItem } });
    expect(loadLanguage()).toBe('en');
    expect(getItem).toHaveBeenCalledWith(LANGUAGE_STORAGE_KEY);
    saveLanguage('zh');
    expect(setItem).toHaveBeenCalledExactlyOnceWith(LANGUAGE_STORAGE_KEY, 'zh');
    getItem.mockReturnValue('invalid');
    expect(loadLanguage()).toBe('zh');
    vi.stubGlobal('window', { get localStorage() { throw new Error('storage unavailable'); } });
    expect(loadLanguage()).toBe('zh');
    expect(() => saveLanguage('en')).not.toThrow();
  });
});
