import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { workbenchEnglish } from './locales/workbench';
import { panelEnglish } from './locales/panels';
import { diagnosticEnglish } from './locales/diagnostics';

export type Language = 'zh' | 'en';
type Values = Record<string, string | number>;
export const LANGUAGE_STORAGE_KEY = 'tensorlab-language';
export const englishMessages: Record<string, string> = { ...workbenchEnglish, ...diagnosticEnglish, ...panelEnglish };
const placeholder = /\{([a-zA-Z]\w*)\}/g;
const templates = Object.entries(englishMessages).filter(([source]) => /\{[a-zA-Z]\w*\}/.test(source)).map(([source, english]) => {
  const names: string[] = [];
  const parts = source.split(/(\{[a-zA-Z]\w*\})/g).map(part => {
    if (/^\{[a-zA-Z]\w*\}$/.test(part)) { names.push(part.slice(1, -1)); return '([\\s\\S]+?)'; }
    return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  });
  return { pattern: new RegExp(`^${parts.join('')}$`), names, english };
});
const interpolate = (text: string, values: Values) => text.replace(placeholder, (token, name: string) => Object.hasOwn(values, name) ? String(values[name]) : token);

export function translate(language: Language, source: string, values: Values = {}, depth = 0): string {
  if (language === 'zh') return interpolate(source, values);
  if (Object.hasOwn(englishMessages, source)) return interpolate(englishMessages[source], values);
  // Existing async status/diagnostic strings remain locale-neutral in state.
  // Only known templates are translated; arbitrary server/user text is preserved.
  if (depth < 4 && /[㐀-鿿]/.test(source)) {
    for (const { pattern, names, english } of templates) {
      const match = source.match(pattern);
      if (match) return interpolate(english, Object.fromEntries(names.map((name, index) => [name, translate(language, match[index + 1], {}, depth + 1)])));
    }
  }
  return interpolate(source, values);
}

export function loadLanguage(): Language {
  try { return typeof window !== 'undefined' && window.localStorage.getItem(LANGUAGE_STORAGE_KEY) === 'en' ? 'en' : 'zh'; }
  catch { return 'zh'; }
}
export function saveLanguage(language: Language) {
  try { if (typeof window !== 'undefined') window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language); }
  catch { /* Language switching still works when browser storage is unavailable. */ }
}

type I18n = { language: Language; setLanguage: (language: Language) => void; t: (source: string, values?: Values) => string };
export const I18nContext = createContext<I18n>({ language: 'zh', setLanguage: () => {}, t: (source, values) => translate('zh', source, values) });
export function I18nProvider({ children, initialLanguage }: { children: ReactNode; initialLanguage?: Language }) {
  const [language, setLanguage] = useState<Language>(() => initialLanguage ?? loadLanguage());
  const t = useCallback((source: string, values?: Values) => translate(language, source, values), [language]);
  const value = useMemo(() => ({ language, setLanguage, t }), [language, t]);
  useEffect(() => {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    saveLanguage(language);
  }, [language]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
export const useI18n = () => useContext(I18nContext);
export function LanguageSwitcher() {
  const { language, setLanguage, t } = useI18n();
  return <select className="language-switcher" aria-label={t('界面语言')} title={t('界面语言')} value={language} onChange={event => setLanguage(event.target.value as Language)}><option value="zh">中文</option><option value="en">English</option></select>;
}
