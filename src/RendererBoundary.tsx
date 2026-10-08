import { Component, type ContextType, type ErrorInfo, type ReactNode } from 'react';
import { I18nContext } from './i18n';
import './rendererViews.css';

type Props = { children: ReactNode; label: string };
type State = { error: Error | null; retry: number };

export default class RendererBoundary extends Component<Props, State> {
  static contextType = I18nContext;
  declare context: ContextType<typeof I18nContext>;
  state: State = { error: null, retry: 0 };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`${this.props.label}渲染失败`, error, info.componentStack);
  }

  private retry = () => this.setState(({ retry }) => ({ error: null, retry: retry + 1 }));

  render() {
    const { t } = this.context;
    if (this.state.error) return <div className="renderer-error" role="alert">
      <strong>{t('{view}暂时无法显示', { view: t(this.props.label) })}</strong>
      <span>{t(this.state.error.message || '渲染器发生未知错误。')}</span>
      <button type="button" className="button subtle" onClick={this.retry}>{t('重试{view}', { view: t(this.props.label) })}</button>
    </div>;
    return <div className="renderer-content" key={this.state.retry}>{this.props.children}</div>;
  }
}
