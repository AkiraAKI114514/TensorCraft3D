import { useMemo } from 'react';
import { Braces, Code2, Download } from 'lucide-react';
import { download, generatePython, safeFilename } from './export';
import type { Graph } from './types';
import { useI18n } from './i18n';

type Props = { graph: Graph; valid: boolean; tab: 'python' | 'json'; onTab: (tab: 'python' | 'json') => void; onExport: () => void };
/** Exporting a structure-only model as Python would hand out code that cannot run on the machine it was built for. */
const STRUCTURE_ONLY_PYTHON = '仅结构模型不能导出可运行的 Python；可以导出 Graph JSON';

export default function CodeExport({ graph, valid, tab, onTab, onExport }: Props) {
  const { t } = useI18n();
  const python = useMemo(() => {
    if (tab !== 'python') return { code: '', error: '' };
    if (graph.structureOnly) return { code: `# ${t(STRUCTURE_ONLY_PYTHON)}`, error: STRUCTURE_ONLY_PYTHON };
    try { return { code: generatePython(graph), error: '' }; }
    catch (reason) { return { code: `# ${(reason as Error).message}`, error: (reason as Error).message }; }
  }, [graph, tab, t]);
  const content = tab === 'python' ? python.code : JSON.stringify(graph, null, 2);
  return <>
    <div className="view-tabs code-tabs">
      <button className={tab === 'python' ? 'active' : ''} onClick={() => onTab('python')}><Code2 size={14} />PyTorch</button>
      <button className={tab === 'json' ? 'active' : ''} onClick={() => onTab('json')}><Braces size={14} />Graph JSON</button>
    </div>
    <pre className="code-preview"><code>{content}</code></pre>
    <div className="modal-footer">
      <span className="muted-text">{graph.structureOnly && tab === 'python' ? t(STRUCTURE_ONLY_PYTHON) : valid ? t('形状校验通过') : t('请先修复结构错误')}</span>
      <button className="button primary" disabled={tab === 'python' && (!valid || !!python.error)} onClick={() => {
        download(content, `${safeFilename(graph.name)}.${tab === 'python' ? 'py' : 'json'}`); onExport();
      }}><Download size={15} />{t('下载')} {tab === 'python' ? '.py' : '.json'}</button>
    </div>
  </>;
}
