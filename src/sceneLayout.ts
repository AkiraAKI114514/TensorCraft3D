import * as THREE from 'three';
import type { Analysis, Edge, Graph, Layer } from './types';
import { attentionLayout, isAttention } from './attentionLayout';
import { crossInputRole, isCrossAttention, isProjectionPort } from './attentionConfig';
import { outerResidualCurve, residualEdges } from './graphRoutes';

export type Dims = [number, number, number];
export type AttentionGeometry = ReturnType<typeof attentionLayout>;

/**
 * One layer as the scene draws it: where its centre is, the box it occupies, its attention geometry
 * if it has one, the row of parallel layers it belongs to and the height its label hangs at.
 */
export interface SceneNode { node: Layer; position: THREE.Vector3; dims: Dims; attention: AttentionGeometry | null; row: number; labelY: number; }
export interface SceneRoute extends Edge { start: THREE.Vector3; end: THREE.Vector3; control1?: THREE.Vector3; control2?: THREE.Vector3; residual: boolean; }
/**
 * The whole scene's geometry, computed once per graph, analysis and spacing. The renderer, the
 * camera framing, the kernel windows and the image export all read it, so every layer's size and
 * every attention module's layout is worked out in one place instead of again by each of them.
 */
export interface SceneLayout {
  nodes: Record<string, SceneNode>; routes: SceneRoute[]; bounds: THREE.Box3; dimensions: Record<string, Dims>;
  /** The typical distance between neighbouring stages, which decides how much a label may say at a zoom. */
  pitch: number;
}

/** How far below the lowest block of its row a label hangs. */
export const LABEL_GAP = 0.6;

export const isMerge = (op: string) => op === 'Add' || op === 'Multiply' || op === 'Divide';

/** The box a non-attention layer is drawn as, sized from the tensor it outputs. */
export function blockDimensions(shape: number[] | undefined, op: string): Dims {
  if (isMerge(op)) return [0.3, 0.3, 0.3];
  // Latent attention draws three parts side by side (input, latent, output), so it is wider; MoE
  // draws its gate, a column of experts, its shared experts and the output, wider still.
  if (op === 'LatentAttention' || op === 'MoE') return [op === 'MoE' ? 1.3 : 1.15, Math.min(2.5, Math.max(0.65, Math.log2((shape?.at(-1) ?? 64) + 1) * 0.25)), 0.8];
  if (!shape) return [0.5, 1, 1];
  if (shape.length === 4) return [Math.max(0.22, Math.log2(shape[1] + 1) * 0.15), Math.max(0.6, Math.log2(shape[2] + 1) * 0.38), Math.max(0.6, Math.log2(shape[3] + 1) * 0.38)];
  return [['ReLU', 'GELU', 'Dropout', 'Flatten', 'Unflatten'].includes(op) ? 0.18 : 0.55, Math.min(2.5, Math.max(0.65, Math.log2(shape.at(-1)! + 1) * 0.25)), 0.8];
}

/** The cubic a scene edge follows; a route with no control points bends a little over its length. */
export function flowCurve({ start, end, control1, control2, compact = false }: { start: THREE.Vector3; end: THREE.Vector3; control1?: THREE.Vector3; control2?: THREE.Vector3; compact?: boolean }) {
  const d = end.x - start.x, bend = compact ? 0.05 : 0.4;
  return new THREE.CubicBezierCurve3(start, control1 ?? start.clone().add(new THREE.Vector3(d * 0.36, bend, 0)), control2 ?? end.clone().add(new THREE.Vector3(-d * 0.36, bend, 0)), end);
}

/** Rank of every node: one more than its deepest parent, in analysis order. */
function ranks(graph: Graph, analysis: Analysis) {
  const parents = new Map<string, string[]>();
  graph.edges.forEach(edge => parents.set(edge.target, [...(parents.get(edge.target) ?? []), edge.source]));
  const rank: Record<string, number> = {};
  for (const id of analysis.order) rank[id] = Math.max(-1, ...(parents.get(id) ?? []).map(source => rank[source] ?? -1)) + 1;
  return rank;
}

/** Height of the first row's axis when nothing in it is taller: a block's centre stands this high. */
const BASE_AXIS = 1.65;
/** How far an attention module's lowest part stands above the ground. */
const ATTENTION_LIFT = 1.05;
/** Room between the tallest layers of two neighbouring rows, for the upper row's labels. */
const ROW_GAP = 1.5;

/**
 * Stages left to right along x, one per rank; a stage is as wide as its widest attention module
 * needs, and parallel nodes of a stage stack upward in rows.
 *
 * Every row has one axis, and every node of the row, attention modules included, is centred on
 * it. Edges enter and leave blocks and modules at their centre height, so the edges along a row
 * run straight instead of climbing to each module and back down. The first row's axis is raised
 * just enough that its tallest module keeps its clearance above the ground; each further row
 * sits above the one below with room for its labels between them.
 */
function placeNodes(graph: Graph, analysis: Analysis, expanded: boolean, dims: Record<string, Dims>, attention: Record<string, AttentionGeometry>) {
  const rank = ranks(graph, analysis);
  const max = Math.max(1, ...Object.values(rank));
  const minimumWidth = graph.nodes.some(n => isAttention(n.op)) ? (expanded ? 4 : 3.5) : (expanded ? 2.8 : 2.1);
  const widths = Array<number>(max + 1).fill(minimumWidth);
  graph.nodes.forEach(node => {
    const r = rank[node.id], module = attention[node.id];
    if (r === undefined || !module || r > max) return;
    widths[r] = Math.max(widths[r], module.dimensions[0] + 1.6);
  });
  const total = widths.reduce((a, b) => a + b, 0);
  const centers = widths.map((w, r) => widths.slice(0, r).reduce((a, b) => a + b, 0) + w / 2 - total / 2);
  const counts: Record<number, number> = {}, rowOf: Record<string, number> = {}, stageOf: Record<string, number> = {};
  graph.nodes.forEach((node, index) => {
    const r = rank[node.id] ?? index, row = counts[r] || 0; counts[r] = row + 1;
    rowOf[node.id] = row; stageOf[node.id] = r;
  });
  const rowCount = Math.max(0, ...Object.values(rowOf)) + 1;
  const half = Array<number>(rowCount).fill(0), lift = Array<number>(rowCount).fill(BASE_AXIS);
  graph.nodes.forEach(node => {
    const row = rowOf[node.id], halfHeight = dims[node.id][1] / 2;
    half[row] = Math.max(half[row], halfHeight);
    if (attention[node.id]) lift[row] = Math.max(lift[row], halfHeight + ATTENTION_LIFT);
  });
  const axes: number[] = [];
  for (let row = 0; row < rowCount; row++) axes.push(row ? Math.max(lift[row], axes[row - 1] + Math.max(3, half[row - 1] + half[row] + ROW_GAP)) : lift[0]);
  const positions: Record<string, THREE.Vector3> = {};
  graph.nodes.forEach(node => {
    const r = stageOf[node.id];
    positions[node.id] = new THREE.Vector3(centers[r] ?? (r - max / 2) * 2.1, axes[rowOf[node.id]], 0);
  });
  const sorted = [...widths].sort((a, b) => a - b);
  return { positions, rowOf, pitch: sorted[Math.floor(sorted.length / 2)] };
}

/** Where an edge leaves or enters a node: an attention module's own port, or the middle of a block's side. */
function port(entry: SceneNode, side: 'input' | 'output' | 'context') {
  const offset = entry.attention ? new THREE.Vector3(...entry.attention.ports[side]) : new THREE.Vector3((side === 'input' ? -1 : 1) * entry.dims[0] / 2, 0, 0);
  return entry.position.clone().add(offset);
}

function routeEdges(graph: Graph, nodes: Record<string, SceneNode>): SceneRoute[] {
  const shortcuts = residualEdges(graph), lanes = new Map<string, number>();
  const boxes = Object.fromEntries(Object.values(nodes).map(entry => [entry.node.id, new THREE.Box3().setFromCenterAndSize(entry.position, new THREE.Vector3(...entry.dims))]));
  return graph.edges.flatMap<SceneRoute>(edge => {
    const source = nodes[edge.source], target = nodes[edge.target];
    if (!source || !target) return [];
    const projection = (entry: SceneNode, name?: string) => entry.attention && name ? entry.attention.projectionPositions[name] : undefined;
    const startPoint = projection(source, edge.sourcePort), endPoint = projection(target, edge.targetPort);
    const start = startPoint ? source.position.clone().add(new THREE.Vector3(...startPoint)) : port(source, 'output');
    const side = !isProjectionPort(edge.targetPort) && isCrossAttention(target.node) && crossInputRole(graph, target.node, edge.id) === 'context' ? 'context' : 'input';
    const end = endPoint ? target.position.clone().add(new THREE.Vector3(...endPoint)) : port(target, side);
    if (!shortcuts.has(edge.id)) return [{ ...edge, start, end, residual: false, control1: undefined, control2: undefined }];
    // A residual shortcut lands on top of its Add and arches over every node in between.
    end.copy(target.position).add(new THREE.Vector3(0, 0.15, 0));
    const obstacles = graph.nodes.filter(n => n.id !== edge.source && n.id !== edge.target && boxes[n.id])
      .map(n => ({ min: boxes[n.id].min.toArray(), max: boxes[n.id].max.toArray() }))
      .filter(box => box.max[0] > start.x && box.min[0] < end.x);
    const lane = lanes.get(edge.target) ?? 0; lanes.set(edge.target, lane + 1);
    const path = outerResidualCurve(start.toArray(), end.toArray(), obstacles, lane);
    return [{ ...edge, start, end, residual: true, control1: new THREE.Vector3(...path.control1), control2: new THREE.Vector3(...path.control2) }];
  });
}

/** Every node's box (with room for its label underneath) and every residual arch. */
function sceneBounds(nodes: Record<string, SceneNode>, routes: SceneRoute[]) {
  const box = new THREE.Box3();
  Object.values(nodes).forEach(entry => {
    const nodeBox = new THREE.Box3().setFromCenterAndSize(entry.position, new THREE.Vector3(...entry.dims));
    // Room for the label: its baseline and a line or two of text under it.
    nodeBox.min.y = Math.min(nodeBox.min.y - 1.1, entry.labelY - 0.5);
    box.union(nodeBox);
  });
  routes.filter(route => route.residual).forEach(route => flowCurve(route).getPoints(40).forEach(point => box.expandByPoint(point)));
  return box.isEmpty() ? new THREE.Box3(new THREE.Vector3(-5, 0, -2), new THREE.Vector3(5, 5, 2)) : box;
}

export function sceneLayout(graph: Graph, analysis: Analysis, expanded: boolean): SceneLayout {
  const attention: Record<string, AttentionGeometry> = {};
  graph.nodes.forEach(node => { if (isAttention(node.op)) attention[node.id] = attentionLayout(node, expanded); });
  const dimensions: Record<string, Dims> = {};
  graph.nodes.forEach(node => { dimensions[node.id] = attention[node.id]?.dimensions as Dims | undefined ?? blockDimensions(analysis.layers[node.id]?.output, node.op); });
  const { positions, rowOf, pitch } = placeNodes(graph, analysis, expanded, dimensions, attention);
  const nodes: Record<string, SceneNode> = {};
  graph.nodes.forEach(node => { nodes[node.id] = { node, position: positions[node.id], dims: dimensions[node.id], attention: attention[node.id] ?? null, row: rowOf[node.id], labelY: 0 }; });
  // The blocks of a row hang their labels on one baseline under the row's lowest block, so labels
  // line up instead of stepping with block heights. An attention module is a tall open frame
  // centred on the same axis; measured against it, the blocks' labels would hang far below them,
  // so it labels its own lower edge instead.
  const baselines = new Map<number, number>(), bottom = (entry: SceneNode) => entry.position.y - entry.dims[1] / 2 - LABEL_GAP;
  Object.values(nodes).forEach(entry => { if (!entry.attention) baselines.set(entry.row, Math.min(baselines.get(entry.row) ?? Infinity, bottom(entry))); });
  Object.values(nodes).forEach(entry => { entry.labelY = entry.attention ? bottom(entry) : baselines.get(entry.row)!; });
  const routes = routeEdges(graph, nodes);
  return { nodes, routes, bounds: sceneBounds(nodes, routes), dimensions, pitch };
}
