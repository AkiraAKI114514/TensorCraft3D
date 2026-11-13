import { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Box, LoaderCircle, TriangleAlert, Upload, X } from 'lucide-react';
import { isGroup, type Graph } from './types';
import { useI18n } from './i18n';
import { tensorNumber, type AttentionSelection, type InferenceReport, type ParameterSnapshot } from './inferenceTypes';
import TensorReadout from './TensorReadout';
import { MAX_OBSERVATION_BYTES, ObservationSnapshotError, parseObservationSnapshot, snapshotMatches, withChannel, type ObservationSnapshot } from './observationSnapshot';
import type { TrainedModelMetadata } from './trainedModel';
import { observationFromReport, observationFromSnapshot, type SceneObservation } from './sceneObservation';
import AttentionInspector from './AttentionInspector';
import ValueExplanationView, { EXPLAINED_OPS, defaultExplainIndex, explainAxes } from './ValueExplanation';
import { analyze } from './analysis';
import { attentionConfig, edgeOutputShape, incomingEdges, isAttention, isProjectionPort } from './attentionConfig';
import { expandGraph, expandedLayerAt, graphAt } from './subgraph';
import './TensorInspector.css';

type Props = { graph: Graph; drillPath?: string[]; trainedModel?: TrainedModelMetadata | null; trainedReady?: boolean; attentionReady?: boolean; selected: string | null; active: boolean; ready: boolean; cuda: boolean; valid: boolean; blocked: boolean; onBusy: (busy: boolean) => void; onObserved?: () => void; onOpenGraph?: (graph: Graph) => void; onSceneObservation?: (observation: SceneObservation) => void };
type Result = { graph: Graph; settings: string; report: InferenceReport };
const ROOT: string[] = [];
const NO_ATTENTION: AttentionSelection = { branch: 0, head: 0, queryStart: 0, keyStart: 0 };

export default function TensorInspector({ graph, drillPath = ROOT, trainedModel = null, trainedReady = false, attentionReady = false, selected, active, ready, cuda, valid, blocked, onBusy, onObserved, onOpenGraph, onSceneObservation }: Props) {
  const { t } = useI18n();
  // `selected` is a layer of the level the views show. Inference, snapshots and the analysis below
  // all use the expanded graph, where a layer inside a folded block exists once per instance; the
  // inspector observes the instance chosen for each block on the drill path (0 by default), and a
  // block node itself as the exit layer of its last instance.
  const [instances, setInstances] = useState<Record<string, number>>({});
  const level = useMemo(() => { try { return graphAt(graph, drillPath); } catch { return graph; } }, [graph, drillPath]);
  const node = level.nodes.find(layer => layer.id === selected);
  const target = useMemo(() => selected ? expandedLayerAt(graph, drillPath, selected, instances) : null, [graph, drillPath, selected, instances]);
  const observedId = target?.id ?? null;
  const runtime = useMemo(() => { try { return expandGraph(graph); } catch { return null; } }, [graph]);
  const observed = observedId ? runtime?.nodes.find(layer => layer.id === observedId) : undefined;
  // 观测来源：在工作台里运行推理，或只读打开外部 Python 进程写出的快照。快照只在内存里，
  // 不写入项目，也不经过后端；它的数值只在当前图与快照的计算图一致时显示。
  const [source, setSource] = useState<'live' | 'offline'>('live');
  const live = source === 'live';
  const [offline, setOffline] = useState<ObservationSnapshot | null>(null), [offlineError, setOfflineError] = useState('');
  const [offlineChannels, setOfflineChannels] = useState<Record<string, number>>({});
  const snapshotFile = useRef<HTMLInputElement>(null);
  const offlineBound = useMemo(() => !!offline && snapshotMatches(graph, offline), [graph, offline]);
  const [weightMode, setWeightMode] = useState<'random' | 'trained'>(trainedModel ? 'trained' : 'random');
  const modelId = weightMode === 'trained' ? trainedModel?.modelId : undefined;
  const weightsAvailable = weightMode === 'random' || (!!modelId && trainedReady);
  useEffect(() => { if (trainedModel) setWeightMode('trained'); }, [trainedModel]);
  const [seed, setSeed] = useState(42), [device, setDevice] = useState<'cpu' | 'cuda'>('cpu');
  const [inputMode, setInputMode] = useState<'synthetic' | 'provided'>('synthetic'), [inputText, setInputText] = useState('');
  const [indices, setIndices] = useState<Record<string, number[]>>({});
  // Weights and biases of the observed layer: which tensor is shown, and its leading indices
  // (filter, input channel), keyed by `${layer}::${tensor}`.
  const [parameterChoice, setParameterChoice] = useState<Record<string, string>>({});
  const [parameterIndices, setParameterIndices] = useState<Record<string, number[]>>({});
  const [offlineParameterChannels, setOfflineParameterChannels] = useState<Record<string, number>>({});
  const [attentionSelections, setAttentionSelections] = useState<Record<string, AttentionSelection>>({});
  // Shapes of what actually runs, keyed by the expanded IDs that inference and snapshots use.
  const layers = useMemo(() => runtime ? analyze(runtime).layers : {}, [runtime]);
  const attentionNode = observed && isAttention(observed.op) ? observed : undefined;
  const attentionConfigValue = attentionNode ? attentionConfig(attentionNode.params) : null;
  const attentionSelection = observedId ? attentionSelections[observedId] ?? NO_ATTENTION : NO_ATTENTION;
  const queryLength = attentionNode ? layers[attentionNode.id]?.output[1] ?? 1 : 1;
  const kvInput = attentionNode && runtime ? runtime.edges.find(edge => edge.target === attentionNode.id && edge.targetPort === `b${attentionSelection.branch}:k0`) ?? incomingEdges(runtime, attentionNode).filter(edge => !isProjectionPort(edge.targetPort))[attentionNode.params.attention_type === 'cross' ? 1 : 0] : undefined;
  const keyLength = kvInput && runtime ? edgeOutputShape(runtime, kvInput, layers)?.[1] ?? 1 : 1;
  const attentionValid = !attentionNode || Object.values(attentionSelection).every(value => Number.isInteger(value) && value >= 0) && attentionSelection.branch < attentionConfigValue!.branches && attentionSelection.head < attentionConfigValue!.heads && attentionSelection.queryStart < queryLength && attentionSelection.keyStart < keyLength;
  const updateAttention = (key: keyof AttentionSelection, value: number) => { if (observedId) setAttentionSelections(previous => ({ ...previous, [observedId]: { ...attentionSelection, [key]: value } })); };
  // One output element of a Linear, convolution or pooling layer to split into its terms; it is
  // part of the run, so changing it needs a rerun like any other selection.
  const [explainIndices, setExplainIndices] = useState<Record<string, number[]>>({});
  const explainShape = observed && EXPLAINED_OPS.has(observed.op) ? layers[observed.id]?.output : undefined;
  const explainIndex = observed && explainShape ? explainIndices[observed.id] ?? defaultExplainIndex(observed.op, explainShape) : [];
  const explainValid = !explainShape || explainIndex.length === explainShape.length - 1 && explainIndex.every((value, axis) => Number.isInteger(value) && value >= 0 && value < explainShape[axis + 1]);
  const changeExplainIndex = (axis: number, value: number) => {
    if (!observedId) return;
    const next = [...explainIndex]; next[axis] = value;
    setExplainIndices(previous => ({ ...previous, [observedId]: next }));
  };
  // One backward pass of the output's cross-entropy against a chosen class; classifiers only.
  const [gradientOn, setGradientOn] = useState(false), [gradientTarget, setGradientTarget] = useState(0);
  const outputId = graph.nodes.find(layer => layer.op === 'Output')?.id;
  const outputShape = outputId ? layers[outputId]?.output : undefined;
  const classes = outputShape?.length === 2 && outputShape[1] >= 2 && outputShape[1] <= 256 ? outputShape[1] : 0;
  const gradientRequested = gradientOn && classes > 0;
  const gradientValid = !gradientRequested || Number.isInteger(gradientTarget) && gradientTarget >= 0 && gradientTarget < classes;
  const chooseInstance = (key: string, index: number) => setInstances(previous => ({ ...previous, [key]: index }));
  const [result, setResult] = useState<Result | null>(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const requesting = useRef(false), controller = useRef<AbortController | null>(null), mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; controller.current?.abort(); }; }, []);
  useEffect(() => { setIndices({}); setAttentionSelections({}); setParameterIndices({}); setExplainIndices({}); setGradientTarget(0); setError(''); }, [graph]);
  const settings = JSON.stringify({ seed, device, weightMode, modelId, inputMode, inputText: inputMode === 'provided' ? inputText : '', indices, attentionSelections, parameterIndices, explainIndices, gradient: gradientRequested ? gradientTarget : null });
  const current = result?.graph === graph && result.settings === settings && weightsAvailable;
  const previousSnapshot = result?.graph === graph && observedId ? result.report.tensors.find(tensor => tensor.nodeId === observedId) : undefined;
  const offlineTensor = offlineBound && offline && observedId ? offline.tensors.find(tensor => tensor.nodeId === observedId) : undefined;
  const offlineChannel = offlineTensor ? offlineChannels[offlineTensor.nodeId] : undefined;
  const snapshot = live ? (current ? previousSnapshot : undefined) : offlineTensor && (offlineChannel === undefined ? offlineTensor : withChannel(offlineTensor, offlineChannel));
  const attentionSnapshot = !observedId ? undefined : live ? (current ? result.report.attentions?.find(value => value.nodeId === observedId) : undefined) : offlineBound && offline ? offline.attentions.find(value => value.nodeId === observedId) : undefined;
  const explanation = live && current && observedId && result.report.explanation?.nodeId === observedId ? result.report.explanation : undefined;
  const gradientReport = live && current ? result.report.gradient ?? undefined : undefined;
  const gradientSnapshot = gradientReport && observedId ? gradientReport.tensors.find(tensor => tensor.nodeId === observedId) : undefined;
  // What the views below show comes from one run or one file; a new one remounts them.
  const viewKey = live ? result?.report.runId ?? '' : offline?.snapshotId ?? '';
  // The observed layer's weights and biases: shown from a current run or a bound snapshot; the
  // last run's list also drives the controls, so they stay while a changed index awaits a rerun.
  const lastParameters: ParameterSnapshot[] = observedId && result?.graph === graph ? (result.report.parameters ?? []).filter(entry => entry.nodeId === observedId) : [];
  const parameterList: ParameterSnapshot[] = !observedId ? [] : live ? (current ? lastParameters : []) : offlineBound && offline ? offline.parameters[observedId] ?? [] : [];
  const parameterOptions = live ? lastParameters : parameterList;
  const parameterName = observedId && parameterOptions.some(entry => entry.name === parameterChoice[observedId]) ? parameterChoice[observedId] : parameterOptions[0]?.name;
  const parameterControl = parameterOptions.find(entry => entry.name === parameterName);
  const parameterKey = observedId && parameterName ? `${observedId}::${parameterName}` : '';
  const parameterShown = parameterList.find(entry => entry.name === parameterName);
  const parameter = parameterShown && !live && offlineParameterChannels[parameterKey] !== undefined ? withChannel(parameterShown, offlineParameterChannels[parameterKey]) : parameterShown;
  const parameterOwner = parameter?.owner ? runtime?.nodes.find(layer => layer.id === parameter.owner)?.name ?? parameter.owner : null;
  const parameterGradient = gradientReport && parameter ? gradientReport.parameters.find(entry => entry.name === parameter.name) : undefined;
  const changeParameterIndex = (axis: number, value: number) => {
    if (!parameterControl || !parameterKey) return;
    const next = [...(parameterIndices[parameterKey] ?? parameterControl.slice.indices)]; next[axis] = value;
    setParameterIndices(previous => ({ ...previous, [parameterKey]: next }));
  };
  const openSnapshot = async (file: File) => {
    setOfflineError('');
    try {
      if (file.size > MAX_OBSERVATION_BYTES) throw new ObservationSnapshotError('观测快照不能超过 16 MB');
      const parsed = parseObservationSnapshot(await file.text());
      if (mounted.current) { setOffline(parsed); setOfflineChannels({}); }
    } catch (reason) {
      if (mounted.current) setOfflineError(reason instanceof ObservationSnapshotError ? reason.message : '无法读取观测快照');
    }
  };
  // `everyLayer` observes every executed layer at once and hands the result to the 3D view; it
  // needs no selected layer and takes no attention window.
  const run = async (everyLayer = false) => {
    if (requesting.current || blocked || !ready || !valid || !weightsAvailable || (!everyLayer && (!observedId || !attentionValid || !explainValid || !gradientValid))) return;
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
      // The runtime expands folded blocks, so the request names the expanded layer.
      const nodeIds = everyLayer ? [] : Array.from(new Set([observedId!, output]));
      const sliced = everyLayer ? Object.entries(indices).slice(0, 8) : nodeIds.filter(id => indices[id]).map(id => [id, indices[id]] as const);
      // The observed layer's weights come along: the same model, so the same weights the values used.
      const ownParameters = observedId ? Object.entries(parameterIndices).filter(([key]) => key.startsWith(`${observedId}::`)).slice(0, 8).map(([key, value]) => [key.slice(observedId.length + 2), value] as const) : [];
      const weights = observedId ? { parameters: observedId, parameterSlices: Object.fromEntries(ownParameters) } : {};
      const response = await fetch('/api/infer', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.current.signal,
        body: JSON.stringify(everyLayer
          ? { graph, allNodes: true, seed, device, modelId, inputs, slices: Object.fromEntries(sliced), ...weights }
          : { graph, nodeIds, seed, device, modelId, inputs, attention: attentionNode && attentionReady ? { [attentionNode.id]: attentionSelection } : undefined, slices: Object.fromEntries(sliced), ...weights, explain: explainShape ? { nodeId: observedId, index: explainIndex } : undefined, gradient: gradientRequested ? { target: gradientTarget } : undefined })
      });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `单样本推理失败 (${response.status})`);
      if (modelId ? body.weights !== 'trained' || body.model?.modelId !== modelId : body.weights !== 'random-initialized') throw new Error('推理返回的权重来源与请求不一致，请更新并重启后端');
      if (!everyLayer && attentionNode && attentionReady && !body.attentions?.some((value: { nodeId: string; branch: number; head: number }) => value.nodeId === attentionNode.id && value.branch === attentionSelection.branch && value.head === attentionSelection.head)) throw new Error('推理响应缺少所选 Attention 观测，请更新并重启后端');
      if (mounted.current) {
        setResult({ graph, settings, report: body as InferenceReport }); onObserved?.();
        if (everyLayer) onSceneObservation?.(observationFromReport(body as InferenceReport, graph));
      }
    } catch (reason) {
      if (mounted.current && (reason as Error).name !== 'AbortError') { setResult(null); setError((reason as Error).message || '无法连接推理服务'); }
    } finally {
      requesting.current = false; if (mounted.current) setBusy(false); onBusy(false);
    }
  };
  const changeIndex = (axis: number, value: number) => {
    if (!previousSnapshot || !observedId) return;
    const next = [...(indices[observedId] ?? previousSnapshot.slice.indices)]; next[axis] = value;
    setIndices(previous => ({ ...previous, [observedId]: next }));
  };
  // Keep the pending request mounted when another inspector tab is selected.
  return <section className="tensor-inspector" aria-label={t('Tensor Inspector')} hidden={!active} aria-busy={busy}>
    <div className="tensor-heading"><span className="eyebrow">OBSERVATION / SINGLE SAMPLE</span><h2>{t('真实张量观测')}</h2></div>
    <label className="field-label">{t('观测来源')}<select aria-label={t('观测来源')} value={source} disabled={busy} onChange={e => setSource(e.target.value as 'live' | 'offline')}><option value="live">{t('在工作台运行单样本推理')}</option><option value="offline">{t('外部 Python 观测快照 · 只读')}</option></select></label>
    {node && (drillPath.length > 0 || isGroup(node)) && <div className="tensor-block-scope">
      {target ? <>
        {target.scopes.filter(scope => scope.count > 1).map(scope => <label className="field-label" key={scope.key}>{t('结构块实例 · {block}', { block: scope.label })}<input aria-label={t('观测实例 · {block}', { block: scope.label })} type="number" min={0} max={scope.count - 1} step={1} value={scope.index} disabled={busy} onChange={e => chooseInstance(scope.key, Number(e.target.value))} /><span>{t('共 {count} 个实例，编号从 0 开始', { count: scope.count })}</span></label>)}
        <p className="tensor-note">{isGroup(node) ? t('结构块的输出来自最后一个实例的出口层 {id}。', { id: target.id }) : t('观测展开后的层 {id}。', { id: target.id })}</p>
      </> : <p className="tensor-note">{t('无法确定这一层在展开后的模型中的位置。')}</p>}
    </div>}
    {live && <>
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
    {observed && explainShape && <div className="value-explain-controls"><h3>{t('解释一个输出值')}</h3>
      <div className="tensor-prefix-controls">{explainAxes(observed.op, explainShape.length).map((name, axis) => {
        const label = name ? t(name) : t('输出轴 {axis}', { axis: axis + 1 });
        return <label className="field-label" key={axis}>{label}<input aria-label={t('要解释的值 · {axis}', { axis: label })} type="number" min={0} max={explainShape[axis + 1] - 1} step={1} value={explainIndex[axis]} disabled={busy} onChange={e => changeExplainIndex(axis, Number(e.target.value))} /></label>;
      })}</div>
      <p className="tensor-note">{t('运行时把这个输出元素拆成实测的各项：Linear 与卷积是输入 × 权重再加偏置，池化是窗口内的平均或最大值。修改位置后需重新运行。')}</p>
    </div>}
    {classes > 0 && <div className="gradient-controls"><h3>{t('单样本梯度')}</h3>
      <label className="checkbox-label"><input type="checkbox" checked={gradientOn} disabled={busy} onChange={e => setGradientOn(e.target.checked)} />{t('同时反向传播一次')}</label>
      {gradientOn && <label className="field-label">{t('目标类别')}<select aria-label={t('梯度目标类别')} value={gradientTarget} disabled={busy} onChange={e => setGradientTarget(Number(e.target.value))}>{Array.from({ length: classes }, (_, index) => <option key={index} value={index}>{index}</option>)}</select></label>}
      <p className="tensor-note">{t('对输出 logits 与所选类别的交叉熵做一次反向传播，显示这一层输出与权重的梯度。eval 模式：dropout 关闭，BatchNorm 用运行统计；不更新权重，也不保留计算图。序列的梯度按位置显示，不是 token 梯度。')}</p>
    </div>}
    <button className="button primary wide" disabled={busy || blocked || !ready || !valid || !weightsAvailable || !attentionValid || !explainValid || !gradientValid || !observedId || !Number.isInteger(seed) || seed < 0 || seed > 2147483647} onClick={() => void run()}>{busy ? <LoaderCircle size={14} className="spin" /> : <Activity size={14} />}{busy ? t('单样本推理中') : t('运行单样本推理')}</button>
    {onSceneObservation && <button className="button subtle wide" disabled={busy || blocked || !ready || !valid || !weightsAvailable || !Number.isInteger(seed) || seed < 0 || seed > 2147483647} onClick={() => void run(true)}><Box size={14} />{t('观测全部层，画到三维视图')}</button>}
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
    </>}
    {!live && <>
      <p className="tensor-note">{t('外部 Python 进程用 backend/observation_snapshot.py 写出结构和采样张量；这里只读显示，不运行推理，也不读取权重文件。')}</p>
      <input ref={snapshotFile} type="file" accept=".json,application/json" hidden aria-label={t('观测快照文件')} onChange={e => { const file = e.target.files?.[0]; e.target.value = ''; if (file) void openSnapshot(file); }} />
      <div className="tensor-snapshot-actions">
        <button className="button subtle" onClick={() => snapshotFile.current?.click()}><Upload size={14} />{offline ? t('换一个观测快照') : t('打开观测快照')}</button>
        {offline && <button className="button subtle" onClick={() => { setOffline(null); setOfflineChannels({}); setOfflineError(''); }}><X size={14} />{t('关闭快照')}</button>}
      </div>
      {offlineError && <div className="diagnostic-item error" role="alert"><TriangleAlert size={15} /><p>{t(offlineError)}</p></div>}
      {offline && <div className="tensor-provenance tensor-offline">
        <strong>{t('离线快照 · {device} · eval', { device: offline.device })}</strong>
        {offline.label && <span>{offline.label}</span>}
        <span>{t('来源：{source} · 样本 {sample} · 种子 {seed}', { source: offline.inputSource === 'synthetic' ? t('合成样本') : t('自定义样本'), sample: offline.sampleIndex, seed: offline.seed })}</span>
        <span>{t('权重：{weights}', { weights: offline.weights === 'external' ? t('外部进程加载的权重') : t('随机初始化（不是已训练模型）') })}</span>
        <time dateTime={offline.createdAt}>{offline.createdAt}</time>
        <code title={offline.snapshotId}>Snapshot {offline.snapshotId}</code>
      </div>}
      {offline && !offlineBound && <div className="tensor-stale" role="status"><p>{t('快照测量的是另一张计算图，数值不能套用到当前模型。')}</p>{onOpenGraph && <button className="button subtle" disabled={blocked} onClick={() => onOpenGraph(offline.graph)}>{t('打开快照中的模型')}</button>}</div>}
      {offlineBound && offline && onSceneObservation && <button className="button subtle wide" onClick={() => onSceneObservation(observationFromSnapshot(offline))}><Box size={14} />{t('把快照画到三维视图')}</button>}
      {offlineBound && <p className="tensor-note">{node ? t('显示 {name} 的离线观测。切片和 Attention 窗口固定在采集时的位置；通道网格可以在已采集的通道之间切换。', { name: node.name }) : t('请在三维视图或拓扑图中选择一层。')}</p>}
    </>}
    {attentionSnapshot && <AttentionInspector key={viewKey + attentionSnapshot.nodeId} snapshot={attentionSnapshot} transformer={observed?.op === 'Transformer'} onSelectHead={live && !busy && attentionReady ? head => updateAttention('head', head) : undefined} />}
    {live && current && !snapshot && <p className="tensor-note">{t('当前层未包含在本次采样中，选择该层后重新运行。')}</p>}
    {!live && offlineBound && node && !snapshot && <p className="tensor-note">{t('快照里没有这一层的观测。')}</p>}
    {snapshot && <div className="tensor-activation"><TensorReadout title={`${node?.name} · ${t('实测输出')}`} snapshot={snapshot} viewKey={viewKey + snapshot.nodeId} tableLabel={t('实测张量切片')} disabled={live && busy} rerun={live} onSelectChannel={channel => live ? changeIndex(1, channel) : setOfflineChannels(previous => ({ ...previous, [snapshot.nodeId]: channel }))} /></div>}
    {explanation && <ValueExplanationView key={viewKey + explanation.nodeId} explanation={explanation} layer={node?.name ?? explanation.nodeId} input={runtime?.nodes.find(layer => layer.id === explanation.inputNode)?.name ?? explanation.inputNode} />}
    {gradientReport && observedId && <div className="tensor-gradient">
      <div className="tensor-gradient-summary" role="status"><strong>{t('单样本梯度 · 交叉熵对类别 {target}', { target: gradientReport.target })}</strong><span>{t('损失 {loss} · 预测类别 {predicted} · 目标类别概率 {probability}', { loss: t(tensorNumber(gradientReport.loss)), predicted: gradientReport.predicted, probability: gradientReport.probability === null ? t('非有限值') : `${(gradientReport.probability * 100).toFixed(1)}%` })}</span></div>
      {gradientSnapshot
        ? <TensorReadout title={`${node?.name} · ${t('输出的梯度 ∂L/∂y')}`} snapshot={gradientSnapshot} viewKey={`${viewKey}gradient${gradientSnapshot.nodeId}`} tableLabel={t('梯度切片')} disabled={busy} rerun onSelectChannel={channel => changeIndex(1, channel)} />
        : <p className="tensor-note">{t('这一层的输出没有梯度：它是整数 ID 输入，或不在通往输出的计算路径上。')}</p>}
    </div>}
    {parameterOptions.length > 0 && <div className="tensor-parameters">
      <div className="inspector-section"><h3>{t('权重与偏置')}<span>{live ? (result?.report.weights === 'trained' ? t('本地训练') : t('随机初始化（不是已训练模型）')) : offline?.weights === 'external' ? t('外部进程加载的权重') : t('随机初始化（不是已训练模型）')}</span></h3>
        <label className="field-label">{t('参数张量')}<select aria-label={t('查看的参数张量')} value={parameterName} disabled={busy} onChange={e => { if (observedId) setParameterChoice(previous => ({ ...previous, [observedId]: e.target.value })); }}>{parameterOptions.map(entry => <option key={entry.name} value={entry.name}>{`${entry.name} · [${entry.shape.join(', ')}]`}</option>)}</select></label>
        {parameterOwner && <p className="tensor-note">{t('这一层共享 {owner} 的权重，显示的是同一组参数。', { owner: parameterOwner })}</p>}
        {live && parameterControl && parameterControl.slice.indices.length > 0 && <div className="tensor-prefix-controls">{parameterControl.slice.indices.map((value, axis) => <label className="field-label" key={axis}>{t(['滤波器', '输入通道', '深度'][axis])}<input aria-label={t('参数切片轴 {axis}', { axis })} type="number" min={0} max={parameterControl.shape[axis] - 1} value={parameterIndices[parameterKey]?.[axis] ?? value} disabled={busy} onChange={e => changeParameterIndex(axis, Number(e.target.value))} /></label>)}</div>}
        <p className="tensor-note">{t('来自这次运行所用的同一个模型。修改滤波器或输入通道后需重新运行。')}</p>
      </div>
      {parameter && <TensorReadout title={t('{name} · 参数', { name: parameter.name })} snapshot={parameter} subject="parameter" viewKey={`${viewKey}${parameterKey}`} tableLabel={t('参数切片')} disabled={live && busy} rerun={live} onSelectChannel={channel => live ? changeParameterIndex(1, channel) : setOfflineParameterChannels(previous => ({ ...previous, [parameterKey]: channel }))} />}
      {parameterGradient && <div className="tensor-gradient"><TensorReadout title={t('{name} · 梯度 ∂L/∂W', { name: parameterGradient.name })} snapshot={parameterGradient} subject="parameter" viewKey={`${viewKey}${parameterKey}gradient`} tableLabel={t('参数梯度切片')} disabled={busy} rerun onSelectChannel={channel => changeParameterIndex(1, channel)} /></div>}
    </div>}
  </section>;
}
