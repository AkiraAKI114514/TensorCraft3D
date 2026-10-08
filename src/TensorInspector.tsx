import { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, LoaderCircle, TriangleAlert } from 'lucide-react';
import type { Graph } from './types';
import { useI18n } from './i18n';
import type { AttentionSelection, InferenceReport } from './inferenceTypes';
import { tensorNumber } from './inferenceTypes';
import type { TrainedModelMetadata } from './trainedModel';
import TensorHistogram from './TensorHistogram';
import AttentionInspector from './AttentionInspector';
import { analyze } from './analysis';
import { attentionConfig, edgeOutputShape, incomingEdges, isAttention, isProjectionPort } from './attentionConfig';
import './TensorInspector.css';

type Props = { graph: Graph; trainedModel?: TrainedModelMetadata | null; trainedReady?: boolean; attentionReady?: boolean; selected: string | null; active: boolean; ready: boolean; cuda: boolean; valid: boolean; blocked: boolean; onBusy: (busy: boolean) => void };
type Result = { graph: Graph; settings: string; report: InferenceReport };

export default function TensorInspector({ graph, trainedModel = null, trainedReady = false, attentionReady = false, selected, active, ready, cuda, valid, blocked, onBusy }: Props) {
  const { t } = useI18n();
  const [weightMode, setWeightMode] = useState<'random' | 'trained'>(trainedModel ? 'trained' : 'random');
  const modelId = weightMode === 'trained' ? trainedModel?.modelId : undefined;
  const weightsAvailable = weightMode === 'random' || (!!modelId && trainedReady);
  useEffect(() => { if (trainedModel) setWeightMode('trained'); }, [trainedModel]);
  const [seed, setSeed] = useState(42), [device, setDevice] = useState<'cpu' | 'cuda'>('cpu');
  const [inputMode, setInputMode] = useState<'synthetic' | 'provided'>('synthetic'), [inputText, setInputText] = useState('');
  const [indices, setIndices] = useState<Record<string, number[]>>({});
  const [attentionSelections, setAttentionSelections] = useState<Record<string, AttentionSelection>>({});
  const layers = useMemo(() => analyze(graph).layers, [graph]);
  const node = graph.nodes.find(layer => layer.id === selected);
  const attentionNode = node && isAttention(node.op) ? node : undefined;
  const attentionConfigValue = attentionNode ? attentionConfig(attentionNode.params) : null;
  const attentionSelection = selected ? attentionSelections[selected] ?? { branch: 0, head: 0, queryStart: 0, keyStart: 0 } : { branch: 0, head: 0, queryStart: 0, keyStart: 0 };
  const queryLength = attentionNode ? layers[attentionNode.id]?.output[1] ?? 1 : 1;
  const kvInput = attentionNode ? graph.edges.find(edge => edge.target === attentionNode.id && edge.targetPort === `b${attentionSelection.branch}:k0`) ?? incomingEdges(graph, attentionNode).filter(edge => !isProjectionPort(edge.targetPort))[attentionNode.params.attention_type === 'cross' ? 1 : 0] : undefined;
  const keyLength = kvInput ? edgeOutputShape(graph, kvInput, layers)?.[1] ?? 1 : 1;
  const attentionValid = !attentionNode || Object.values(attentionSelection).every(value => Number.isInteger(value) && value >= 0) && attentionSelection.branch < attentionConfigValue!.branches && attentionSelection.head < attentionConfigValue!.heads && attentionSelection.queryStart < queryLength && attentionSelection.keyStart < keyLength;
  const updateAttention = (key: keyof AttentionSelection, value: number) => { if (selected) setAttentionSelections(previous => ({ ...previous, [selected]: { ...attentionSelection, [key]: value } })); };
  const [result, setResult] = useState<Result | null>(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const requesting = useRef(false), controller = useRef<AbortController | null>(null), mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; controller.current?.abort(); }; }, []);
  useEffect(() => { setIndices({}); setAttentionSelections({}); setError(''); }, [graph]);
  const settings = JSON.stringify({ seed, device, weightMode, modelId, inputMode, inputText: inputMode === 'provided' ? inputText : '', indices, attentionSelections });
  const current = result?.graph === graph && result.settings === settings && weightsAvailable;
  const previousSnapshot = result?.graph === graph ? result.report.tensors.find(tensor => tensor.nodeId === selected) : undefined;
  const snapshot = current ? previousSnapshot : undefined;
  const attentionSnapshot = current ? result.report.attentions?.find(value => value.nodeId === selected) : undefined;
  const run = async () => {
    if (requesting.current || blocked || !ready || !valid || !selected || !weightsAvailable || !attentionValid) return;
    requesting.current = true; setBusy(true); setError(''); onBusy(true);
    controller.current = new AbortController();
    try {
      let inputs: Record<string, number[]> | undefined;
      if (inputMode === 'provided') {
        const value: unknown = JSON.parse(inputText);
        const ids = graph.nodes.filter(layer => layer.op === 'Input').map(layer => layer.id);
        if (!value || Array.isArray(value) || typeof value !== 'object' || Object.keys(value).length !== ids.length || ids.some(id => !(id in value))) throw new Error('输入 JSON 必须包含所有 Input 层 ID，且不能包含额外 ID');
        inputs = value as Record<string, number[]>;
        for (const id of ids) {
          const shape = graph.nodes.find(layer => layer.id === id)!.params.shape as number[];
          if (!Array.isArray(inputs[id]) || inputs[id].length !== shape.slice(1).reduce((a, b) => a * b, 1) || inputs[id].some(v => typeof v !== 'number' || !Number.isFinite(v))) throw new Error(`${id} 需要一个样本的有限数值平铺数组`);
        }
      }
      const output = graph.nodes.find(layer => layer.op === 'Output')!.id;
      const nodeIds = Array.from(new Set([selected, output]));
      const response = await fetch('/api/infer', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.current.signal,
        body: JSON.stringify({ graph, nodeIds, seed, device, modelId, inputs, attention: attentionNode && attentionReady ? { [attentionNode.id]: attentionSelection } : undefined, slices: Object.fromEntries(nodeIds.filter(id => indices[id]).map(id => [id, indices[id]])) })
      });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `单样本推理失败 (${response.status})`);
      if (modelId ? body.weights !== 'trained' || body.model?.modelId !== modelId : body.weights !== 'random-initialized') throw new Error('推理返回的权重来源与请求不一致，请更新并重启后端');
      if (attentionNode && attentionReady && !body.attentions?.some((value: { nodeId: string; branch: number; head: number }) => value.nodeId === attentionNode.id && value.branch === attentionSelection.branch && value.head === attentionSelection.head)) throw new Error('推理响应缺少所选 Attention 观测，请更新并重启后端');
      if (mounted.current) setResult({ graph, settings, report: body as InferenceReport });
    } catch (reason) {
      if (mounted.current && (reason as Error).name !== 'AbortError') { setResult(null); setError((reason as Error).message || '无法连接推理服务'); }
    } finally {
      requesting.current = false; if (mounted.current) setBusy(false); onBusy(false);
    }
  };
  const changeIndex = (axis: number, value: number) => {
    if (!previousSnapshot || !selected) return;
    const next = [...(indices[selected] ?? previousSnapshot.slice.indices)]; next[axis] = value;
    setIndices(previous => ({ ...previous, [selected]: next }));
  };
  // Keep the pending request mounted when another inspector tab is selected.
  return <section className="tensor-inspector" aria-label={t('Tensor Inspector')} hidden={!active} aria-busy={busy}>
    <div className="tensor-heading"><span className="eyebrow">OBSERVATION / SINGLE SAMPLE</span><h2>{t('真实张量观测')}</h2></div>
    <p className="tensor-note">{t('单样本 eval 推理 · 已训练权重或固定种子随机初始化权重；不加载导入源码的原始权重。粒子动画仍是结构示意。')}</p>
    {!ready && <p className="tensor-note">{t('需要支持单样本推理的 Python / PyTorch 服务，请更新并重启后端。')}</p>}
    <label className="field-label">{t('模型权重')}<select aria-label={t('推理权重来源')} value={weightMode} disabled={busy} onChange={e => setWeightMode(e.target.value as 'random' | 'trained')}><option value="trained">{t('本地训练权重')}</option><option value="random">{t('随机初始化权重')}</option></select></label>
    {weightMode === 'trained' && (trainedModel ? <div className="tensor-provenance tensor-trained-model">
      <strong>{t('本地训练模型 · {device} · {reason}', { device: trainedModel.device, reason: trainedModel.reason === 'early_stopping' ? t('早停最佳权重') : t('最终权重') })}</strong>
      <span>{t('数据：{dataset} · 训练 {epochs} 轮 · 权重轮次 {weightsEpoch} · 训练种子 {seed}', { dataset: trainedModel.dataset === 'csv' ? 'CSV' : t('合成分类数据'), epochs: trainedModel.epochsCompleted, weightsEpoch: trainedModel.weightsEpoch, seed: trainedModel.seed })}</span>
      <code>Model {trainedModel.modelId}</code><code>Training {trainedModel.trainingRunId}</code>
      <time dateTime={trainedModel.createdAt}>{trainedModel.createdAt}</time>
    </div> : <p className="tensor-stale">{t('当前计算图没有已确认的训练权重。请完成真实训练或恢复对应图；也可显式选择随机初始化。')}</p>)}
    {weightMode === 'trained' && !trainedReady && <p className="tensor-note">{t('后端暂不支持训练权重推理，请更新并重启。')}</p>}
    <p className="tensor-note">{t('只保留后端内存中的最近一次成功训练（最多 64 MiB）；重启或其他页面的新训练可能使快照失效。停止、失败和演示不生成训练模型。')}</p>
    <div className="tensor-controls">
      <label className="field-label">{weightMode === 'trained' ? t('样本随机种子（不改变训练权重）') : t('随机种子')}<input aria-label={t('推理随机种子')} type="number" min={0} max={2147483647} step={1} value={seed} disabled={busy} onChange={e => setSeed(Number(e.target.value))} /></label>
      <label className="field-label">{t('推理设备')}<select aria-label={t('推理设备')} value={device} disabled={busy} onChange={e => setDevice(e.target.value as 'cpu' | 'cuda')}><option value="cpu">CPU</option><option value="cuda" disabled={!cuda}>CUDA</option></select></label>
    </div>
    <label className="field-label">{t('输入样本')}<select aria-label={t('推理输入来源')} value={inputMode} disabled={busy} onChange={e => setInputMode(e.target.value as 'synthetic' | 'provided')}><option value="synthetic">{t('固定种子合成样本')}</option><option value="provided">{t('自定义单样本 JSON')}</option></select></label>
    {inputMode === 'provided' && <label className="field-label">{t('输入 ID → 平铺数组')}<textarea aria-label={t('推理输入 JSON')} value={inputText} disabled={busy} placeholder={JSON.stringify(Object.fromEntries(graph.nodes.filter(layer => layer.op === 'Input').map(layer => [layer.id, []])))} onChange={e => setInputText(e.target.value)} /><span>{t('每个输入省略 batch 轴，按原维度顺序平铺；Embedding 输入使用合法整数 ID。')}{weightMode === 'trained' && trainedModel?.preprocessing === 'csv-standardized' ? t('输入原始 CSV 数值，自动应用训练集拟合的标准化。') : t('不应用 CSV 标准化。')}</span></label>}
    {previousSnapshot && <div className="tensor-prefix-controls">{previousSnapshot.slice.indices.map((value, axis) => <label className="field-label" key={axis}>{t('切片轴 {axis}', { axis })}{axis === 0 ? ` · ${t('样本')}` : ''}<input aria-label={t('张量切片轴 {axis}', { axis })} type="number" min={0} max={previousSnapshot.shape[axis] - 1} value={indices[previousSnapshot.nodeId]?.[axis] ?? value} disabled={busy || previousSnapshot.shape[axis] === 1} onChange={e => changeIndex(axis, Number(e.target.value))} /></label>)}</div>}
    {attentionNode && <div className="attention-observation"><h3>{t('Attention 采样窗口')}</h3>
      <div className="attention-filters">
        <label className="field-label">{t('分支')}<select aria-label={t('Attention 分支')} value={attentionSelection.branch} disabled={busy || !attentionReady} onChange={event => updateAttention('branch', Number(event.target.value))}>{Array.from({ length: Math.max(1, Math.min(8, Math.floor(attentionConfigValue!.branches) || 1)) }, (_, index) => <option value={index} key={index}>B{index + 1}</option>)}</select></label>
        <label className="field-label">Query Head<select aria-label={t('Attention Head')} value={attentionSelection.head} disabled={busy || !attentionReady} onChange={event => updateAttention('head', Number(event.target.value))}>{Array.from({ length: Math.max(1, Math.min(16, Math.floor(attentionConfigValue!.heads) || 1)) }, (_, index) => <option value={index} key={index}>H{index + 1}</option>)}</select></label>
        <label className="field-label">{t('Query 起点')}<input aria-label={t('Attention Query 起点')} type="number" min={0} max={queryLength - 1} value={attentionSelection.queryStart} disabled={busy || !attentionReady} onChange={event => updateAttention('queryStart', Number(event.target.value))} /></label>
        <label className="field-label">{t('Key 起点')}<input aria-label={t('Attention Key 起点')} type="number" min={0} max={keyLength - 1} value={attentionSelection.keyStart} disabled={busy || !attentionReady} onChange={event => updateAttention('keyStart', Number(event.target.value))} /></label>
      </div><p className="tensor-note">{t('只采样所选分支与 Head，每个矩阵最多 16 × 16；修改窗口后重新运行，完整矩阵统计不变。')}{!attentionReady && t('后端暂不支持 Attention 内部观测；当前仅采集层输出。')}</p>
    </div>}
    <button className="button primary wide" disabled={busy || blocked || !ready || !valid || !weightsAvailable || !attentionValid || !node || !Number.isInteger(seed) || seed < 0 || seed > 2147483647} onClick={() => void run()}>{busy ? <LoaderCircle size={14} className="spin" /> : <Activity size={14} />}{busy ? t('单样本推理中') : t('运行单样本推理')}</button>
    <p className="tensor-note">{node ? t('采样 {name} 与模型输出；修改切片后需重新运行。', { name: node.name }) : t('请在三维视图或拓扑图中选择一层。')}{t('训练或环境检查期间不可运行。')}</p>
    {error && <div className="diagnostic-item error" role="alert"><TriangleAlert size={15} /><p>{t(error)}</p></div>}
    {result && !current && <p className="tensor-stale" role="status">{t('模型或采样配置已变化，旧快照已失效，请重新运行。')}</p>}
    {busy && result && <p className="tensor-note" role="status">{t('正在采样；以下若有快照，仍来自上一次运行。')}</p>}
    {current && <div className="tensor-provenance">
      <strong>{t('实测快照 · {device} · eval', { device: result.report.device })}</strong>
      <span>{t('来源：{source} · 样本 {sample} · 种子 {seed}', { source: result.report.inputSource === 'synthetic' ? t('合成样本') : t('自定义样本'), sample: result.report.sampleIndex, seed: result.report.seed })}</span>
      <span>{t('权重：{weights}', { weights: result.report.weights === 'trained' ? t('本地训练') : t('随机初始化（不是已训练模型）') })}</span>
      <span>{t('预处理：{transform}', { transform: result.report.inputTransform === 'csv-standardized' ? t('训练集 CSV 标准化') : t('无') })}</span>
      {result.report.model && <><code>Model {result.report.model.modelId}</code><code>Training {result.report.model.trainingRunId}</code><span>{t('权重轮次 {weightsEpoch} · {reason}', { weightsEpoch: result.report.model.weightsEpoch, reason: result.report.model.reason === 'early_stopping' ? t('早停最佳权重') : t('最终权重') })}</span><code title={result.report.model.graphFingerprint}>Graph {result.report.model.graphFingerprint}</code></>}
      <time dateTime={result.report.createdAt}>{result.report.createdAt}</time>
      <code title={result.report.runId}>Run {result.report.runId}</code>
    </div>}
    {attentionSnapshot && result && <AttentionInspector key={result.report.runId + attentionSnapshot.nodeId} snapshot={attentionSnapshot} transformer={node?.op === 'Transformer'} />}
    {current && !snapshot && <p className="tensor-note">{t('当前层未包含在本次采样中，选择该层后重新运行。')}</p>}
    {snapshot && <>
      <div className="inspector-section"><h3>{node?.name} · {t('实测输出')}</h3>
        <dl className="tensor-stats">
          <div><dt>Shape</dt><dd>{`[${snapshot.shape.join(', ')}]`}</dd></div><div><dt>Dtype</dt><dd>{snapshot.dtype}</dd></div>
          <div><dt>{t('元素数')}</dt><dd>{snapshot.elements}</dd></div><div><dt>{t('非有限元素')}</dt><dd>{snapshot.nonFiniteCount}</dd></div>
          <div><dt>Min</dt><dd>{t(tensorNumber(snapshot.stats.min))}</dd></div><div><dt>Max</dt><dd>{t(tensorNumber(snapshot.stats.max))}</dd></div>
          <div><dt>Mean</dt><dd>{t(tensorNumber(snapshot.stats.mean))}</dd></div><div><dt>{t('Std · 总体')}</dt><dd>{t(tensorNumber(snapshot.stats.std))}</dd></div>
        </dl>
        {snapshot.nonFiniteCount > 0 && <p className="tensor-stale">{t('存在 NaN/Inf；统计和分布只包含有限元素，切片中以“非有限值”标注。')}</p>}
      </div>
      <TensorHistogram key={result!.report.runId + snapshot.nodeId} histogram={snapshot.histogram} />
      <div className="inspector-section"><h3>{t('张量切片')}<span>{t('最多 16 × 16')}</span></h3>
        <p className="tensor-note">{t('固定前置轴，显示末尾{count}维 {shape} 的左上区域。', { count: snapshot.slice.shape.length, shape: `[${snapshot.slice.shape.join(', ')}]` })}{snapshot.slice.truncated ? t('已截断，表格不是完整张量。') : t('该切片已完整显示。')}</p>
        <div className="tensor-table-scroll" tabIndex={0} role="region" aria-label={t('实测张量切片')}>
          <table className="tensor-slice"><thead><tr><th>{t('索引')}</th>{Array.from({ length: snapshot.slice.columns }, (_, column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{snapshot.slice.values.map((row, i) => <tr key={i}><th>{i}</th>{row.map((value, j) => <td key={j} title={`[${[...snapshot.slice.indices, ...(snapshot.slice.shape.length > 1 ? [i] : []), j].join(', ')}] = ${value === null ? t('非有限值') : value}`}>{t(tensorNumber(value))}</td>)}</tr>)}</tbody></table>
        </div>
      </div>
    </>}
  </section>;
}
