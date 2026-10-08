import { useCallback, useEffect, useRef, useState } from 'react';
import type { Graph, Metric, TrainingConfig } from './types';
import { computationKey, type TrainedModelResult } from './trainedModel';

type StartOptions = { graph: Graph; config: TrainingConfig; valid: boolean; ready: boolean; blocked: boolean };

export default function useTrainingSession(notify: (message: string) => void) {
  const [metrics, setMetrics] = useState<Metric[]>([]), [training, setTraining] = useState(false), [trainStatus, setTrainStatus] = useState('未开始');
  const [trainedResult, setTrainedResult] = useState<TrainedModelResult | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  useEffect(() => () => { const ws = socketRef.current; socketRef.current = null; ws?.close(); }, []);
  const isTrainingActive = useCallback(() => socketRef.current !== null, []);
  const resetMetrics = useCallback(() => { setMetrics([]); setTrainStatus('未开始'); }, []);
  const importMetrics = useCallback((records: Metric[]) => { setMetrics(records); setTrainStatus('指标已导入'); }, []);
  const finishDemo = useCallback(() => setTrainStatus('演示完成'), []);
  const startTraining = ({ graph, config, valid, ready, blocked }: StartOptions) => {
    if (socketRef.current || blocked) return false;
    if (!valid) { notify('模型结构存在错误，请先修复'); return false; }
    if (!ready) { notify('真实训练需要 Python 服务与 PyTorch，运行 install.ps1 -Training'); return false; }
    if (config.dataset === 'csv' && !config.csv) { notify('请先选择 CSV 数据集'); return false; }
    const graphKey = computationKey(graph);
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/train`); socketRef.current = ws;
    setMetrics([]); setTrainedResult(null); setTraining(true); setTrainStatus('准备数据');
    let terminal = false;
    ws.onopen = () => { if (socketRef.current === ws) ws.send(JSON.stringify({ graph, config })); };
    ws.onmessage = event => {
      if (socketRef.current !== ws || terminal) return;
      const message = JSON.parse(event.data);
      if (message.type === 'metric') { setMetrics(m => [...m, message.metric]); setTrainStatus(`训练中 · ${message.device}`); }
      else if (message.type === 'done') {
        terminal = true; socketRef.current = null; setTraining(false);
        if (message.model && ['completed', 'early_stopping'].includes(message.reason)) setTrainedResult({ graphKey, metadata: message.model });
        setTrainStatus(message.reason === 'early_stopping' ? '早停完成' : message.reason === 'stopped' ? '已停止' : '训练完成');
        notify(message.retentionWarning || (message.model ? '训练权重已保留，可在张量观测中推理' : '训练已结束')); ws.close();
      } else if (message.type === 'error') { terminal = true; socketRef.current = null; setTraining(false); setTrainStatus('训练失败'); notify(message.message); ws.close(); }
    };
    ws.onerror = () => { if (socketRef.current !== ws || terminal) return; terminal = true; socketRef.current = null; setTraining(false); setTrainStatus('连接失败'); notify('训练服务连接失败，请检查 Python 服务'); ws.close(); };
    ws.onclose = () => { if (socketRef.current !== ws) return; socketRef.current = null; setTraining(false); if (!terminal) { setTrainStatus('连接中断 · 未收到训练结果'); notify('训练连接中断，未确认本次模型权重'); } };
    return true;
  };
  const stopTraining = () => { const ws = socketRef.current; if (!ws) return; if (ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify({ type: 'stop' })); setTrainStatus('正在停止'); } else ws.close(); };
  return { metrics, setMetrics, training, trainStatus, setTrainStatus, importMetrics, finishDemo, trainedResult, startTraining, stopTraining, isTrainingActive, resetMetrics };
}
