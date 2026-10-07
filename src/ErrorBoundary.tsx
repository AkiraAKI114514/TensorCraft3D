import { Component, type ErrorInfo, type ReactNode } from 'react';
import { download } from './export';
import './ErrorBoundary.css';

type State = { error: Error | null; backupStatus: string };

export default class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, backupStatus: '' };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('TensorCraft3D 工作台发生错误', error, info.componentStack);
  }

  private backupProject = () => {
    try {
      const saved = localStorage.getItem('tensorlab-project');
      if (!saved) { this.setState({ backupStatus: '没有可备份的已保存项目。' }); return; }
      download(saved, 'tensorcraft3d-recovery.json', 'application/json');
      this.setState({ backupStatus: '已下载最近一次成功自动保存的项目。' });
    } catch {
      this.setState({ backupStatus: '无法读取或下载已保存项目，请检查浏览器的存储与下载权限。' });
    }
  };

  render() {
    if (!this.state.error) return this.props.children;
    return <main className="app-error-screen">
      <section className="app-error-card" role="alert" aria-labelledby="app-error-title">
        <span className="eyebrow">TENSORCRAFT3D / RECOVERY</span>
        <h1 id="app-error-title">工作台发生错误</h1>
        <p>工作台暂时无法继续显示。可以先备份已保存的项目，再重新加载页面。</p>
        <p>备份仅包含最近一次成功自动保存的项目，未保存的修改可能无法恢复。重新加载不会删除已保存的项目。</p>
        <div className="app-error-actions">
          <button type="button" className="button subtle" onClick={this.backupProject}>下载已保存项目</button>
          <button type="button" className="button primary" onClick={() => window.location.reload()}>重新加载工作台</button>
        </div>
        {this.state.backupStatus && <p role="status">{this.state.backupStatus}</p>}
        <details className="app-error-details"><summary>错误详情</summary><pre>{this.state.error.message}</pre></details>
      </section>
    </main>;
  }
}
