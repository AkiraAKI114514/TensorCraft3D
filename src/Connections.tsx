import { useState } from 'react';
import { Plus, X } from 'lucide-react';
import type { Connection } from '@xyflow/react';
import type { Graph, Layer } from './types';
import { isCrossAttention, crossInputRole, isProjectionPort, projectionPorts, projectionLabel } from './attentionConfig';

export default function Connections({ graph, node, port = null, disabled, onConnect, onRemove, onRole }: { graph: Graph; node: Layer; port?: string | null; disabled: boolean; onConnect: (connection: Connection) => void; onRemove: (id: string) => void; onRole: (id: string, role: string) => void }) {
  const [input, setInput] = useState(''), [output, setOutput] = useState(''), [sourcePort, setSourcePort] = useState(''), [targetPort, setTargetPort] = useState('');
  const [role, setRole] = useState<'query' | 'context' | ''>('');
  const cross = isCrossAttention(node) && !port;
  const newRole = role || (graph.edges.some(e => e.target === node.id && !isProjectionPort(e.targetPort) && crossInputRole(graph, node, e.id) === 'query') ? 'context' : 'query');
  const projection = projectionPorts(node).find(p => p.id === port);
  const inherited = projection ? graph.edges.filter(e => e.target === node.id && !isProjectionPort(e.targetPort)).find(e => !isCrossAttention(node) || crossInputRole(graph, node, e.id) === (projection.role === 'Q' ? 'query' : 'context')) : undefined;
  return <div className="inspector-section connection-editor"><h3>{port ? `${projectionLabel(node, port)} 连接对象` : '连接对象'}</h3>
    {(['input', 'output'] as const).map(side => {
      const incoming = side === 'input', label = incoming ? '输入' : '输出';
      const edges = graph.edges.filter(e => (incoming ? e.target : e.source) === node.id && (incoming ? port ? e.targetPort === port : !isProjectionPort(e.targetPort) : (e.sourcePort ?? null) === port));
      const value = incoming ? input : output, setValue = incoming ? setInput : setOutput;
      const options = graph.nodes.filter(n => n.id !== node.id);
      const choice = options.some(n => n.id === value) ? value : '', other = options.find(n => n.id === choice);
      const otherPorts = other ? projectionPorts(other) : [];
      const handle = incoming ? sourcePort : targetPort;
      return <div className="connection-group" key={side}>
        <span className="connection-title">{label}对象 · {edges.length}</span>
        {incoming && port && !edges.length && <span className="connection-default">默认 ← {graph.nodes.find(n => n.id === inherited?.source)?.name ?? '未连接'}</span>}
        {edges.map(e => {
          const other = graph.nodes.find(n => n.id === (incoming ? e.source : e.target))!;
          return <div className="connection-row" key={e.id}><span>{incoming ? '←' : '→'}</span><span>{other.name}{(incoming ? e.sourcePort : e.targetPort) ? ` · ${projectionLabel(other, incoming ? e.sourcePort : e.targetPort)}` : ''}</span>{incoming && cross && <select aria-label={`输入角色 ${other.name}`} disabled={disabled} value={crossInputRole(graph, node, e.id)} onChange={event => onRole(e.id, event.target.value)}><option value="query">Query</option><option value="context">Context</option></select>}<button className="icon-button" aria-label={`取消${label}对象 ${other.name}`} disabled={disabled} onClick={() => onRemove(e.id)}><X size={12} /></button></div>;
        })}
        {incoming && cross && <label className="field-label">新增输入角色<select aria-label="新增输入角色" disabled={disabled} value={newRole} onChange={e => setRole(e.target.value as 'query' | 'context')}><option value="query">Query</option><option value="context">Context</option></select></label>}
        <div className="connection-add"><select aria-label={`选择${label}对象`} value={choice} disabled={disabled} onChange={e => { setValue(e.target.value); incoming ? setSourcePort('') : setTargetPort(''); }}><option value="">选择{label}对象</option>{options.map(n => <option value={n.id} key={n.id}>{n.name} · {n.op}</option>)}</select><button className="icon-button" aria-label={`添加${label}连接`} disabled={disabled || !choice} onClick={() => { onConnect({ source: incoming ? choice : node.id, target: incoming ? node.id : choice, sourceHandle: incoming ? sourcePort || null : port, targetHandle: incoming ? port || (cross ? newRole : null) : targetPort || null }); setValue(''); setRole(''); }}><Plus size={14} /></button></div>
        {other && (otherPorts.length > 0) && <label className="field-label">{incoming ? '来源输出端口' : '目标输入端口'}<select aria-label={incoming ? '来源输出端口' : '目标输入端口'} disabled={disabled} value={handle} onChange={e => incoming ? setSourcePort(e.target.value) : setTargetPort(e.target.value)}><option value="">{incoming ? '模型层输出' : '模型层输入'}</option>{!incoming && isCrossAttention(other) && <><option value="query">Query</option><option value="context">Context</option></>}{otherPorts.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}</select></label>}
      </div>;
    })}
  </div>;
}
