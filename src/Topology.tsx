import { useEffect, useMemo, useState } from 'react';
import { ReactFlow, Background, Controls, MiniMap, Panel, Handle, Position, BaseEdge, useUpdateNodeInternals, type ReactFlowInstance, type Edge, type EdgeProps, type NodeProps, type Node, type Connection, type NodeChange, type EdgeChange, MarkerType } from '@xyflow/react';
import { Layers3, Plus, Trash2, Ungroup } from 'lucide-react';
import '@xyflow/react/dist/style.css';
import { COLORS, repeatOf, type Graph, type Analysis } from './types';
import { shapeText } from './analysis';
import { residualEdges } from './graphRoutes';
import { RESIDUAL_COLOR } from './flowGeometry';
import { crossInputRole, isCrossAttention, isProjectionPort, projectionPorts, projectionLabel } from './attentionConfig';
import { useI18n } from './i18n';
type LayerNode = Node<{ name: string; op: string; color: string; shape: string; error: boolean; cross: boolean; repeat: number; ports: ReturnType<typeof projectionPorts> }, 'layer'>;
function Layer({ id, data, selected }: NodeProps<LayerNode>) {
  const updateInternals = useUpdateNodeInternals();
  useEffect(() => updateInternals(id), [id, data.ports.length, data.cross, updateInternals]);
  return <div className={`flow-layer ${selected ? 'selected' : ''} ${data.error ? 'error' : ''}`} style={{ borderTopColor: data.color }}>
    {data.op !== 'Input' && (data.cross ? <><Handle id="query" type="target" position={Position.Left} style={{ top: '32%', background: '#319cac' }} /><Handle id="context" type="target" position={Position.Left} style={{ top: '75%', background: '#a17cbb' }} /></> : <Handle type="target" position={Position.Left} />)}
    <span className="node-op">{data.op}</span><strong>{data.name}</strong>{data.repeat > 1 && <span className="repeat-badge">×{data.repeat}</span>}<code>{data.shape}</code>
    {data.cross && <small className="cross-ports">Q · Query / KV · Context</small>}
    {data.ports.length > 0 && <div className="projection-handles">{data.ports.map(port => <div key={port.id} className="projection-handle"><Handle id={port.id} type="target" position={Position.Left} /><span>{port.label}</span><Handle id={port.id} type="source" position={Position.Right} /></div>)}</div>}
    {data.op !== 'Output' && <Handle type="source" position={Position.Right} />}
  </div>;
}
const nodeTypes = { layer: Layer };
function ResidualEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, selected, data }: EdgeProps<Edge<{ laneY: number }>>) {
  const laneY = data?.laneY ?? Math.min(sourceY, targetY) - 100;
  return <g data-flow-kind="residual"><BaseEdge id={id} path={`M${sourceX},${sourceY} C${sourceX},${laneY} ${targetX},${laneY} ${targetX},${targetY}`} markerEnd={markerEnd} style={{ stroke: RESIDUAL_COLOR, strokeWidth: selected ? 2.5 : 1.5 }} /></g>;
}
const edgeTypes = { residual: ResidualEdge };
export default function Topology({ graph, analysis, selected, onSelect, onChange, onConnect, onAddObject, onRemoveSelected, onGroup, onDissolve, disabled }: { graph: Graph; analysis: Analysis; selected: string | null; onSelect: (id: string | null) => void; onChange: (g: Graph) => void; onConnect: (c: Connection) => void; onAddObject: (op: 'Input' | 'Output') => void; onRemoveSelected: () => void; onGroup: (members: string[], name?: string) => void; onDissolve: (id: string) => void; disabled: boolean }) {
  const { t } = useI18n();
  const [instance, setInstance] = useState<ReactFlowInstance<LayerNode> | null>(null);
  // 手动分组需要多选：Shift 框选或按住 Shift 点选，选中的 id 由 React Flow 汇报。
  const [picked, setPicked] = useState<string[]>([]);
  const selectedGroup = selected ? graph.nodes.find(n => n.id === selected && n.op === 'Group') : undefined;
  useEffect(() => { void instance?.fitView({ padding: 0.2, duration: 160 }); }, [instance, graph.nodes.length]);
  const nodes = useMemo(() => graph.nodes.map(n => ({ id: n.id, type: 'layer' as const, position: n.position, selected: n.id === selected, data: { name: n.name, op: n.op, color: COLORS[n.op], shape: shapeText(analysis.layers[n.id]?.output), error: analysis.diagnostics.some(d => d.level === 'error' && d.nodeId === n.id), cross: isCrossAttention(n), repeat: repeatOf(n), ports: projectionPorts(n) } })), [graph, analysis, selected]);
  const shortcuts = residualEdges(graph);
  const edges = graph.edges.map(e => {
    const residual = shortcuts.has(e.id), source = graph.nodes.find(n => n.id === e.source), target = graph.nodes.find(n => n.id === e.target);
    const between = graph.nodes.filter(n => source && target && n.position.x >= Math.min(source.position.x, target.position.x) && n.position.x <= Math.max(source.position.x, target.position.x));
    const color = residual ? RESIDUAL_COLOR : analysis.layers[e.target] ? '#8299a3' : '#db6666';
    const role = isProjectionPort(e.targetPort) ? e.targetPort : target && isCrossAttention(target) ? crossInputRole(graph, target, e.id) : undefined;
    const label = role === 'query' ? 'Query → Q' : role === 'context' ? 'Context → K/V' : isProjectionPort(role) ? `→ ${projectionLabel(target!, role)}` : e.sourcePort ? `${projectionLabel(source!, e.sourcePort)} →` : undefined;
    return { ...e, sourceHandle: e.sourcePort, targetHandle: role, label, type: residual ? 'residual' : 'default', data: { laneY: Math.min(0, ...between.map(n => n.position.y)) - 150 }, markerEnd: { type: MarkerType.ArrowClosed, color }, style: { stroke: color } };
  });
  const changes = (changes: NodeChange[]) => {
    if (disabled) return;
    let updated = graph;
    for (const c of changes) {
      if (c.type === 'position' && c.position) updated = { ...updated, nodes: updated.nodes.map(n => n.id === c.id ? { ...n, position: c.position! } : n) };
      if (c.type === 'remove') updated = { ...updated, nodes: updated.nodes.filter(n => n.id !== c.id), edges: updated.edges.filter(e => e.source !== c.id && e.target !== c.id) };
    }
    if (updated !== graph) onChange(updated);
  };
  const edgeChanges = (changes: EdgeChange[]) => { if (disabled) return; const deleted = changes.filter(c => c.type === 'remove').map(c => c.id); if (deleted.length) onChange({ ...graph, edges: graph.edges.filter(e => !deleted.includes(e.id)) }); };
  return <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} onInit={setInstance} onNodesChange={changes} onEdgesChange={edgeChanges} onConnect={onConnect} onNodeClick={(_, n) => onSelect(n.id)} onPaneClick={() => onSelect(null)} onSelectionChange={({ nodes: pickedNodes }) => setPicked(pickedNodes.map(n => n.id))} nodesDraggable={!disabled} nodesConnectable={!disabled} fitView minZoom={0.15} maxZoom={2} deleteKeyCode={disabled ? null : 'Delete'} proOptions={{ hideAttribution: true }}>
    <Panel position="top-left" className="topology-toolbar">
      <div className="topology-actions"><button className="button subtle" disabled={disabled} onClick={() => onAddObject('Input')}><Plus size={13} />{t('添加输入对象')}</button><button className="button subtle" disabled={disabled} onClick={() => onAddObject('Output')}><Plus size={13} />{t('添加输出对象')}</button><button className="button subtle" disabled={disabled || !selected} onClick={onRemoveSelected}><Trash2 size={13} />{t('删除选中对象')}</button><span className="toolbar-separator" /><button className="button subtle" disabled={disabled || picked.length < 2} onClick={() => onGroup(picked)}><Layers3 size={13} />{t('组成结构块 · {count}', { count: picked.length })}</button><button className="button subtle" disabled={disabled || !selectedGroup} onClick={() => selectedGroup && onDissolve(selectedGroup.id)}><Ungroup size={13} />{t('解散结构块')}</button></div>
      <span>{t('Shift 框选或点选多个节点后组成结构块；结构块可进入查看内部。箭头连线自由连线；错误原因见右侧诊断。')}</span>
    </Panel>
    <Background gap={20} color="#d0dce1" /><Controls /><MiniMap nodeColor={n => String(n.data.color)} pannable zoomable />
  </ReactFlow>;
}
