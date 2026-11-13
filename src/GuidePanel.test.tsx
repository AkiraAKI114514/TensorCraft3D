import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import GuidePanel from './GuidePanel';
import { I18nProvider } from './i18n';
import { GUIDE_STEPS, completeStep, emptyGuide, guideProgress, nextGuideStep, parseGuide, type GuideState } from './guideState';

const render = (state: GuideState, english = false) => {
  const panel = <GuidePanel state={state} onAction={() => {}} onClose={() => {}} onReset={() => {}} />;
  return renderToStaticMarkup(english ? <I18nProvider initialLanguage="en">{panel}</I18nProvider> : panel);
};

describe('getting-started guide state', () => {
  it('starts empty and treats anything it does not recognise in storage as not done', () => {
    expect(parseGuide(null)).toEqual(emptyGuide());
    for (const raw of ['', 'not json', '[]', 'null', '42', '{"done":"all"}', '{"done":[true]}']) expect(parseGuide(raw)).toEqual(emptyGuide());
    // 只有认识的步骤和真正的 true 会被保留：旧版本或手改过的内容不能把进度凭空点满。
    expect(parseGuide('{"seen":true,"done":{"template":true,"train":"yes","inspect":1,"unknown":true,"export":true}}')).toEqual({ seen: true, done: { template: true, export: true } });
    expect(parseGuide('{"seen":"yes","done":{}}').seen).toBe(false);
  });

  it('completes steps idempotently and in any order', () => {
    const first = completeStep(emptyGuide(), 'inspect');
    expect(first.done).toEqual({ inspect: true });
    // 已完成的步骤返回同一个对象，调用方靠它跳过重复写盘。
    expect(completeStep(first, 'inspect')).toBe(first);
    expect(guideProgress(first)).toBe(1);
    // 乱序完成后，下一步仍是顺序里第一个没做的。
    expect(nextGuideStep(first)).toBe('template');
    const all = GUIDE_STEPS.reduce(completeStep, emptyGuide());
    expect(guideProgress(all)).toBe(GUIDE_STEPS.length);
    expect(nextGuideStep(all)).toBeNull();
  });
});

describe('getting-started guide panel', () => {
  it('lists the five steps, marks the next one, and offers one navigation button per unfinished step', () => {
    const markup = render(emptyGuide());
    expect(markup).toContain('aria-label="上手引导"');
    expect(markup).toContain('已完成 0 / 5 步');
    expect(markup.match(/<li/g)).toHaveLength(5);
    expect(markup.match(/aria-current="step"/g)).toHaveLength(1);
    expect(markup).toMatch(/<li class="current" aria-current="step">.*?选一个模板/);
    for (const action of ['去选模板', '回到三维视图', '打开张量观测', '打开训练设置', '打开代码导出']) expect(markup).toContain(action);
    expect(markup).toContain('不会替你运行推理或训练');
    // 没有进度时「重新开始」没有意义。
    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>重新开始/);
  });

  it('says a step is done in words, not only with the check mark', () => {
    const markup = render({ seen: true, done: { template: true, inspect: true } });
    expect(markup).toContain('已完成 2 / 5 步');
    expect(markup.match(/class="guide-done-label"/g)).toHaveLength(2);
    expect(markup).toMatch(/<li class="current" aria-current="step">.*?点一层看它的形状/);
    expect(markup).not.toMatch(/<button[^>]*disabled=""[^>]*>重新开始/);
  });

  it('closes the list with a finished message once every step is done', () => {
    const markup = render({ seen: true, done: Object.fromEntries(GUIDE_STEPS.map(step => [step, true])) });
    expect(markup).toContain('五步都走完了');
    expect(markup).not.toContain('aria-current');
  });

  it('translates every title, description and button', () => {
    for (const state of [emptyGuide(), { seen: true, done: { template: true } }, { seen: true, done: Object.fromEntries(GUIDE_STEPS.map(step => [step, true])) }]) {
      const markup = render(state, true);
      expect(markup).not.toMatch(/[㐀-鿿]/);
      expect(markup).toContain('Getting started');
    }
  });
});
