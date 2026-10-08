import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { analyze } from './analysis';
import { PRESETS } from './presets';
import { I18nProvider, translate } from './i18n';
import RendererBoundary from './RendererBoundary';
import SceneView from './SceneView';
import TopologyView from './TopologyView';

describe('localized renderer isolation', () => {
  it('translates lazy-loading placeholders without eagerly importing renderers', () => {
    const graph = PRESETS.mlp(), analysis = analyze(graph);
    const markup = renderToStaticMarkup(<I18nProvider initialLanguage="en"><SceneView graph={graph} analysis={analysis} selected={null} selectedHead={null} selectedProjection={null} onSelect={() => {}} onSelectProjection={() => {}} onSelectHead={() => {}} rotating playing direction="forward" speed={1} expanded={false} diagnostics={[]} onReady={() => {}} onGpu={() => {}} /><TopologyView graph={graph} analysis={analysis} selected={null} onSelect={() => {}} onChange={() => {}} onConnect={() => {}} onAddObject={() => {}} onRemoveSelected={() => {}} disabled={false} /></I18nProvider>);
    expect(markup).toContain('Loading 3D renderer…');
    expect(markup).toContain('Loading topology renderer…');
    expect(markup).toContain('aria-live="polite"');
    expect(markup).not.toContain('正在加载');
  });

  it('renders existing renderer failures in the current language without changing retry state', () => {
    const boundary = new RendererBoundary({ label: '三维视图', children: null });
    boundary.state = { error: new Error(''), retry: 3 };
    boundary.context = { language: 'en', setLanguage: () => {}, t: (source, values) => translate('en', source, values) };
    const english = renderToStaticMarkup(boundary.render());
    expect(english).toContain('3D view is temporarily unavailable');
    expect(english).toContain('An unknown renderer error occurred.');
    expect(english).toContain('Retry 3D view');
    boundary.context = { language: 'zh', setLanguage: () => {}, t: (source, values) => translate('zh', source, values) };
    const chinese = renderToStaticMarkup(boundary.render());
    expect(chinese).toContain('三维视图暂时无法显示');
    expect(chinese).toContain('重试三维视图');
    expect(boundary.state.retry).toBe(3);
    expect(boundary.state.error).not.toBeNull();
  });
});
