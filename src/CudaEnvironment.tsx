import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, Copy, ExternalLink, LoaderCircle, RefreshCw, TriangleAlert } from 'lucide-react';
import './CudaEnvironment.css';

interface Diagnosis { code: string; level: 'error' | 'warning' | 'info'; message: string }
interface EnvironmentReport {
  status: string;
  torch: { installed: boolean; version: string | null; cudaBuild: string | null; error?: string };
  cuda: { available: boolean; deviceCount: number; devices: { name: string; index: number }[]; error: string | null };
  gpu: { present: boolean; driver: string | null; devices: { name: string; driver: string; memoryMiB: string }[] };
  interpreter: string;
  environment: Record<string, string>;
  diagnosis: Diagnosis[];
  setup: { pytorch: string; nvidia: string; note: string };
}

export default function CudaEnvironment({ online = true, training = false, active = true, onBusy }: { online?: boolean; training?: boolean; active?: boolean; onBusy?: (busy: boolean) => void }) {
  const requesting = useRef(false);
  const [report, setReport] = useState<EnvironmentReport | null>(null);
  const [busy, setBusy] = useState(false), [smokeBusy, setSmokeBusy] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [smoke, setSmoke] = useState<{ ok: boolean; message: string } | null>(null);
  const refresh = useCallback(async () => {
    if (training || requesting.current) return;
    requesting.current = true; onBusy?.(true);
    setBusy(true); setError(''); setSmoke(null);
    try {
      const response = await fetch('/api/environment');
      if (!response.ok) throw new Error(response.status === 409 ? '训练期间无法检查 CUDA 环境' : `环境检查失败 (${response.status})`);
      setReport(await response.json() as EnvironmentReport);
    } catch (reason) { setError((reason as Error).message || '无法连接 Python 服务'); }
    finally { requesting.current = false; setBusy(false); onBusy?.(false); }
  }, [training, onBusy]);
  useEffect(() => { if (online && active) void refresh(); }, [online, active, refresh]);
  const runSmoke = async () => {
    if (training || requesting.current) return;
    requesting.current = true; onBusy?.(true);
    setSmokeBusy(true); setSmoke(null); setError('');
    try {
      const response = await fetch('/api/environment/smoke', { method: 'POST' });
      const body = await response.json() as { ok?: boolean; device?: string; loss?: number; detail?: string };
      if (!response.ok || !body.ok || !Number.isFinite(body.loss)) throw new Error(body.detail || `GPU 测试失败 (${response.status})`);
      setSmoke({ ok: true, message: `GPU 前向/反向成功 · ${body.device} · loss ${body.loss?.toFixed(5)}` });
    } catch (reason) { setSmoke({ ok: false, message: (reason as Error).message || 'GPU 测试失败' }); }
    finally { requesting.current = false; setSmokeBusy(false); onBusy?.(false); }
  };
  const command = '.\\setup-training.ps1 -Variant cu128';
  const copy = async () => { try { await navigator.clipboard.writeText(command); setCopied(true); setTimeout(() => setCopied(false), 1800); } catch { setError('浏览器不允许复制，请手动选择命令'); } };
  const ready = report?.status === 'available';
  return <section className="cuda-environment" aria-label="CUDA 环境诊断">
    <div className="cuda-environment-heading"><div><span className="eyebrow">RUNTIME / CUDA</span><h2>CUDA 环境诊断</h2></div><button className="button subtle" disabled={!online || busy || smokeBusy || training} onClick={() => void refresh()}>{busy ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}刷新</button></div>
    {!online && <div className="diagnostic-item error"><TriangleAlert size={15} /><p>Python 服务未连接，请先运行 start.ps1。</p></div>}
    {error && <div className="diagnostic-item error" role="alert"><TriangleAlert size={15} /><p>{error}</p></div>}
    {report && <>
      <div className={`backend-notice ${ready ? 'ready' : ''}`}>{ready ? <CheckCircle2 size={17} /> : <TriangleAlert size={17} />}<div><strong>{ready ? 'CUDA 已就绪' : `当前状态：${report.status}`}</strong><p>{report.diagnosis[0]?.message}</p></div></div>
      <div className="cuda-facts"><div><span>PyTorch</span><strong>{report.torch.installed ? report.torch.version : '未安装'}</strong></div><div><span>CUDA build</span><strong>{report.torch.cudaBuild || 'CPU wheel'}</strong></div><div><span>GPU / 驱动</span><strong>{report.gpu.devices[0] ? `${report.gpu.devices[0].name} · ${report.gpu.driver}` : '未检测到'}</strong></div><div><span>解释器</span><code title={report.interpreter}>{report.interpreter}</code></div></div>
      <details className="cuda-details"><summary>诊断详情与环境变量</summary>{report.diagnosis.map(d => <div className={`diagnostic-item ${d.level}`} key={d.code}><TriangleAlert size={14} /><div><strong>{d.code}</strong><p>{d.message}</p></div></div>)}<div className="cuda-vars">{Object.entries(report.environment).map(([key, value]) => <div key={key}><code>{key}</code><span>{value}</span></div>)}{!Object.keys(report.environment).length && <span>未设置已关注的 CUDA 环境变量</span>}</div></details>
      <div className="cuda-setup"><strong>需要安装或切换 wheel？</strong><p>不会自动安装。请在项目目录中手动运行 setup-training.ps1；脚本只修改项目 .venv，不改驱动或环境变量。cu128 使用官方已核实的 Python 3.13 Windows wheel，CPU 可用 -Variant cpu。</p><button className="button subtle" disabled={!online || busy || smokeBusy || training || !report?.cuda.available} onClick={() => void runSmoke()}>{smokeBusy ? <LoaderCircle size={14} className="spin" /> : <CheckCircle2 size={14} />}{smokeBusy ? 'GPU 测试中' : '测试 GPU 前向与反向'}</button>{smoke && <div className={`smoke-result ${smoke.ok ? 'ok' : 'error'}`} role={smoke.ok ? 'status' : 'alert'}>{smoke.message}</div>}<div className="copy-command"><code>{command}</code><button className="icon-button" aria-label="复制 CUDA 安装命令" title="复制命令" onClick={() => void copy()}>{copied ? <CheckCircle2 size={15} /> : <Copy size={15} />}</button></div><div className="cuda-links"><a href={report.setup.pytorch} target="_blank" rel="noreferrer">PyTorch 官方安装选择器 <ExternalLink size={12} /></a><a href={report.setup.nvidia} target="_blank" rel="noreferrer">NVIDIA 驱动文档 <ExternalLink size={12} /></a></div></div>
    </>}
  </section>;
}
