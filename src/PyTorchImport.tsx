import { useRef, useState } from 'react';
import { CheckCircle2, Code2, LoaderCircle, Upload, X } from 'lucide-react';
import { analyze, formatNumber, shapeText, validateGraph } from './analysis';
import type { Graph } from './types';
import { useI18n } from './i18n';

interface ImportDiagnostic { level: 'error' | 'warning' | 'info'; code: string; message: string; line?: number; column?: number; }
interface ImportResult { graph: Graph | null; models: string[]; model: string | null; inputs: { name: string; shape: number[]; inferred: boolean }[]; diagnostics: ImportDiagnostic[]; }
const examples: Record<string, string> = {
  mlp: `import torch\nfrom torch import nn\n\nmodel = nn.Sequential(\n    nn.Linear(16, 32),\n    nn.ReLU(),\n    nn.Dropout(0.2),\n    nn.Linear(32, 4),\n)\n`,
  residual: `import torch\nfrom torch import nn\nimport torch.nn.functional as F\n\nclass ResidualCNN(nn.Module):\n    def __init__(self, classes=10):\n        super().__init__()\n        self.stem = nn.Conv2d(3, 16, 3, padding=1)\n        self.block = nn.Sequential(\n            nn.Conv2d(16, 16, 3, padding=1),\n            nn.BatchNorm2d(16),\n        )\n        self.pool = nn.AdaptiveAvgPool2d(1)\n        self.classifier = nn.Linear(16, classes)\n\n    def forward(self, x):\n        x = F.relu(self.stem(x))\n        residual = x\n        x = F.relu(self.block(x) + residual)\n        x = torch.flatten(self.pool(x), 1)\n        return self.classifier(x)\n`,
  transformer: `import torch\nfrom torch import nn\n\nclass EncoderClassifier(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.encoder = nn.TransformerEncoder(\n            nn.TransformerEncoderLayer(\n                d_model=32, nhead=4, dim_feedforward=64,\n                dropout=0.1, batch_first=True,\n            ),\n            num_layers=2,\n        )\n        self.flatten = nn.Flatten(1)\n        self.classifier = nn.Linear(8 * 32, 4)\n\n    def forward(self, x):\n        return self.classifier(self.flatten(self.encoder(x)))\n`,
  cross: `import torch\nfrom torch import nn\n\nclass CrossAttention(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)\n        self.flatten = nn.Flatten(1)\n        self.classifier = nn.Linear(8 * 32, 4)\n\n    def forward(self, query, context):\n        x, _ = self.attention(query, context, context, need_weights=False)\n        return self.classifier(self.flatten(x))\n`,
};

export default function PyTorchImport({ online, disabled, onClose, onImport }: { online: boolean; disabled: boolean; onClose: () => void; onImport: (graph: Graph) => void }) {
  const { t } = useI18n();
  const [source, setSource] = useState(''), [filename, setFilename] = useState(''), [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ImportResult | null>(null), [model, setModel] = useState('');
  const [activeLine, setActiveLine] = useState<number | null>(null);
  const [shapes, setShapes] = useState<Record<string, string>>({}), [error, setError] = useState('');
  const fileInput = useRef<HTMLInputElement>(null), requestId = useRef(0);
  const invalidate = () => { requestId.current++; setBusy(false); setResult(r => r ? { ...r, graph: null, diagnostics: [] } : null); setError(''); };
  const changeSource = (text: string, name = '') => { invalidate(); setSource(text); setFilename(name); setResult(null); setModel(''); setShapes({}); setActiveLine(null); };
  const sourceLines = source.split(/\r?\n/);
  const lineDiagnostics = new Map<number, ImportDiagnostic>();
  for (const diagnostic of result?.diagnostics ?? []) if (diagnostic.line && !lineDiagnostics.has(diagnostic.line)) lineDiagnostics.set(diagnostic.line, diagnostic);
  const jumpToLine = (line?: number) => { if (!line) return; setActiveLine(line); requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-import-line="${line}"]`)?.scrollIntoView({ block: 'center' })); };
  const upload = async (file: File) => {
    if (file.size > 512_000) { setError('Python 文件不能超过 512 KB'); return; }
    try { changeSource(await file.text(), file.name); } catch { setError('无法读取 Python 文件'); }
  };
  const parse = async () => {
    if (disabled) return;
    const id = ++requestId.current;
    setBusy(true); setError(''); setResult(r => r ? { ...r, graph: null, diagnostics: [] } : null);
    try {
      if (new TextEncoder().encode(source).length > 512_000) throw new Error('代码不能超过 512 KB');
      const input_shapes: Record<string, number[]> = {};
      for (const [name, text] of Object.entries(shapes)) {
        if (!text.trim()) continue;
        const shape = text.split(/[,×x\s]+/).filter(Boolean).map(Number);
        if (![2, 3, 4, 5].includes(shape.length) || shape.some(v => !Number.isInteger(v) || v < 1 || v > 65536)) throw new Error('需要 [B,F]、[B,S,E]、[B,C,H,W] 或 [B,C,D,H,W]，各维度为正整数');
        input_shapes[name] = shape;
      }
      const response = await fetch('/api/import/pytorch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source, model_name: model || null, input_shapes }) });
      if (!response.ok) throw new Error(response.status === 404 || response.status === 405 ? '请重启 Python 服务以启用代码导入接口' : `解析服务返回错误 (${response.status})`);
      const next = await response.json() as ImportResult;
      if (!Array.isArray(next.models) || !Array.isArray(next.diagnostics)) throw new Error('解析服务响应无效，请重启 Python 服务');
      if (next.graph) {
        next.graph = validateGraph(next.graph);
        const check = analyze(next.graph);
        if (!check.valid) throw new Error(`模型校验失败：${check.diagnostics.find(d => d.level === 'error')?.message}`);
      }
      if (id !== requestId.current) return;
      setResult(next); setModel(next.model ?? '');
      setShapes(previous => Object.fromEntries(next.inputs.map(input => [input.name, previous[input.name] ?? input.shape.join(', ')])));
    } catch (e) { if (id === requestId.current) setError((e as Error).message || '解析失败，请检查 Python 服务'); }
    finally { if (id === requestId.current) setBusy(false); }
  };
  const preview = result?.graph ? analyze(result.graph) : null;
  return <div className="modal-overlay" onClick={onClose}>
    <div className="modal pytorch-import-modal" role="dialog" aria-modal="true" aria-label={t('导入 PyTorch 代码')} onClick={e => e.stopPropagation()}>
      <div className="modal-heading"><div><span className="eyebrow">TENSORCRAFT3D / PYTORCH IMPORT</span><h2>{t('导入代码 · 自动建模')}</h2></div><button className="icon-button" aria-label={t('关闭对话框')} onClick={onClose}><X size={18} /></button></div>
      <p className="import-intro">{t('上传或粘贴模型定义，解析层参数与 forward 连接，生成可编辑的三维模型。')}</p>
      <div className="import-source-toolbar"><button className="button subtle" disabled={busy} onClick={() => fileInput.current?.click()}><Upload size={14} />{t('上传 .py')}</button><span title={filename}>{filename || t('或直接粘贴下方')}</span><select aria-label={t('PyTorch 导入示例')} value="" disabled={busy} onChange={e => changeSource(examples[e.target.value])}><option value="" disabled>{t('加载示例')}</option><option value="mlp">Sequential MLP</option><option value="residual">{t('残差 CNN')}</option><option value="transformer">Transformer Encoder</option><option value="cross">Cross-Attention</option></select></div>
      <input ref={fileInput} type="file" accept=".py,text/x-python" hidden onChange={e => { if (e.target.files?.[0]) void upload(e.target.files[0]); e.target.value = ''; }} />
      <div className="import-editor" aria-label={t('PyTorch 源代码')}><pre className="import-line-gutter" aria-hidden="true">{sourceLines.map((_, index) => <span key={index} className={lineDiagnostics.has(index + 1) ? 'error-line' : ''}>{index + 1}</span>)}</pre><div className="import-line-preview" aria-hidden="true">{sourceLines.map((line, index) => <span key={index} data-import-line={index + 1} className={`${lineDiagnostics.has(index + 1) ? 'error-line' : ''} ${activeLine === index + 1 ? 'active-line' : ''}`}>{line || ' '}</span>)}</div><textarea className="import-source" aria-label={t('PyTorch 源代码')} placeholder="from torch import nn\n\nmodel = nn.Sequential(...)" value={source} spellCheck={false} disabled={busy} onChange={e => changeSource(e.target.value, filename)} onScroll={e => { const preview = e.currentTarget.parentElement?.querySelector<HTMLElement>('.import-line-preview'); const gutter = e.currentTarget.parentElement?.querySelector<HTMLElement>('.import-line-gutter'); if (preview) preview.scrollTop = e.currentTarget.scrollTop; if (gutter) gutter.scrollTop = e.currentTarget.scrollTop; }} /></div>
      {result?.models.length ? <label className="field-label">{t('模型对象')}<select aria-label={t('导入模型对象')} value={model} disabled={busy} onChange={e => { invalidate(); setModel(e.target.value); setShapes({}); }} >{result.models.map(name => <option key={name}>{name}</option>)}</select></label> : null}
      {!!result?.inputs.length && <div className="import-inputs"><div className="import-section-title">{t('输入形状')}<small>{t('修改后需重新解析 · 图像 NCHW / 序列 BSE')}</small></div>{result.inputs.map(input => <label className="field-label" key={input.name}>{input.name}<input aria-label={t(`导入输入形状 ${input.name}`)} value={shapes[input.name] ?? ''} disabled={busy} placeholder="1, 3, 32, 32" onChange={e => { invalidate(); setShapes(s => ({ ...s, [input.name]: e.target.value })); }} /></label>)}</div>}
      {!online && <div className="diagnostic-item error">{t('Python 服务未连接，请先运行 start.ps1。代码解析无需安装 PyTorch。')}</div>}
      {error && <div role="alert" className="diagnostic-item error">{t(error)}</div>}
      {result?.diagnostics.map((d, i) => <button key={i} type="button" role={d.level === 'error' ? 'alert' : undefined} className={`diagnostic-item ${d.level}`} onClick={() => jumpToLine(d.line)}><div><strong>{d.line ? t('第 {line} 行{column} · ', { line: d.line, column: d.column ? t(' · 第 {column} 列', { column: d.column }) : '' }) : ''}{d.code}</strong><p>{t(d.message)}</p></div></button>)}
      {result?.graph && preview && <div className="import-preview"><div className="import-section-title"><CheckCircle2 size={14} />{t('解析通过')}<small>{result.graph.nodes.length} {t('个节点 ·')}{result.graph.edges.length} {t('条连接 ·')}{formatNumber(preview.parameters)} {t('参数')}</small></div><div className="import-preview-scroll"><table><thead><tr><th>{t('层名称')}</th><th>{t('类型')}</th><th>{t('输出形状')}</th></tr></thead><tbody>{result.graph.nodes.map(n => <tr key={n.id}><td>{n.name}</td><td>{n.op}</td><td>{shapeText(preview.layers[n.id]?.output)}</td></tr>)}</tbody></table></div></div>}
      <details className="import-support"><summary>{t('支持范围与导入方式')}</summary><p>{t('支持 Sequential、静态 nn.Module、嵌套模块、ModuleDict/ModuleList、残差相加、cat、展平、Conv/ConvTranspose、Linear/Bilinear、Norm、Pool、Dropout、Embedding、Upsample、常用激活、MultiheadAttention、TransformerEncoder、固定位置编码、Slice 和 Select。注意力输入使用 batch_first=True。')}</p><p>{t('只静态解析结构，不执行上传代码；权重和训练脚本不会导入。常量缓冲区保留实际 FP32 数值；切片和位置编码需要明确输入形状。高级索引、batch 轴切片、动态控制流、权重共享、掩码、未知算子或不兼容参数会报出源码位置。可重新导入 TensorCraft3D 导出的 Python。')}</p></details>
      <div className="modal-footer"><button className="button subtle" disabled={busy || !source.trim() || !online || disabled} onClick={() => void parse()}>{busy ? <LoaderCircle size={15} className="spin" /> : <Code2 size={15} />}{busy ? t('正在解析') : t('解析代码')}</button><button className="button primary" disabled={busy || !result?.graph || disabled} onClick={() => { if (result?.graph) onImport(result.graph); }}><CheckCircle2 size={15} />{t('导入模型')}</button></div>
    </div>
  </div>;
}
