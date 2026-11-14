import * as THREE from 'three';
import type { Analysis, Edge, Graph, Layer } from './types';
import { attentionLayout, isAttention } from './attentionLayout';
import { crossInputRole, isCrossAttention, isProjectionPort } from './attentionConfig';
import { outerResidualCurve, residualEdges } from './graphRoutes';

export type Dims = [number, number, number];
export type AttentionGeometry = ReturnType<typeof attentionLayout>;

/**
 * One layer as the scene draws it: where its centre is, the box it occupies, its attention geometry
 * if it has one, the band and the row of parallel layers it belongs to, and the height its label
 * hangs at. Only the lowest band stands on the ground; `grounded` says whether this layer is in it.
 */
export interface SceneNode { node: Layer; position: THREE.Vector3; dims: Dims; attention: AttentionGeometry | null; band: number; row: number; labelY: number; grounded: boolean; }
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
  /** How many bands a long model wraps into, top to bottom; 1 when it fits on one line. */
  bands: number;
}

/** Where the overview camera stands relative to the model's centre; wrapping is judged as it sees the model. */
export const OVERVIEW_OFFSETS = { blocks: [15, 9, 18] as Dims, attention: [14, 8, 26] as Dims };

/** How far below the lowest block of its row a label hangs. */
export const LABEL_GAP = 0.6;

export const isMerge = (op: string) => op === 'Add' || op === 'Multiply' || op === 'Divide';

/** The box a non-attention layer is drawn as, sized from the tensor it outputs. */
export function blockDimensions(shape: number[] | undefined, op: string): Dims {
  if (isMerge(op)) return [0.3, 0.3, 0.3];
  // Latent attention and MoE draw their insides (src/layerAnatomy.ts): the two latents, the heads
  // and their scores; the router, the expert bank, the shared experts and the sum. They need room
  // in all three directions.
  if (op === 'LatentAttention' || op === 'MoE') return [op === 'MoE' ? 2.4 : 2.6, Math.min(2.5, Math.max(1.8, Math.log2((shape?.at(-1) ?? 64) + 1) * 0.25)), op === 'MoE' ? 1.5 : 1.6];
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
/** Height kept under a label's baseline for its text. */
const LABEL_ROOM = 0.9;
/** Height kept over a band whose shortcuts arch above it. */
const ARC_ROOM = 2.8;
/** Gap between one band's labels and the next band's arches or blocks. */
const BAND_GAP = 0.9;
/** The frame the overview fills, for judging how many bands make the model largest. */
const FRAME_ASPECT = 16 / 9;
/** Wrapping must make the overview at least this much larger, or a model stays on one line. */
const WRAP_GAIN = 1.15;
const MAX_BANDS = 6;
/** A model with fewer stages reads better on one line, even where wrapping would draw it a little larger. */
const MIN_WRAP_STAGES = 14;
/** What each edge past the first crossing a break costs, in stages of drift from an even split. */
const CROSSING_COST = 0.6;

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
    if (r === undefined || r > max) return;
    // An attention module keeps its own margin; any other block gets enough room not to touch its
    // neighbours, which only the wide latent attention and MoE blocks need.
    widths[r] = Math.max(widths[r], module ? module.dimensions[0] + 1.6 : dims[node.id][0] + 1.3);
  });
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
  const bandOf = (id: string) => Math.min(stageOf[id], max);
  // How many edges cross the boundary after each stage. A residual stack has a shortcut over
  // nearly every boundary, so a break cannot avoid them all; it prefers boundaries few edges cross.
  const crossings = Array<number>(max).fill(0);
  graph.edges.forEach(edge => {
    const from = stageOf[edge.source], to = stageOf[edge.target];
    if (from === undefined || to === undefined) return;
    for (let r = Math.min(from, to); r < Math.max(from, to) && r < max; r++) crossings[r]++;
  });
  const skips = graph.edges.filter(edge => Math.abs((stageOf[edge.target] ?? 0) - (stageOf[edge.source] ?? 0)) > 1);
  // Places the stages band by band: each band runs left to right from the same left edge, and the
  // bands stack downward, the last one on the ground, each clear of the next one's arches.
  const place = (ranges: [number, number][]) => {
    const band = Array<number>(max + 1).fill(0);
    ranges.forEach(([from, to], index) => { for (let r = from; r < to; r++) band[r] = index; });
    const spans = ranges.map(([from, to]) => widths.slice(from, to).reduce((a, b) => a + b, 0));
    const left = -Math.max(...spans) / 2, x = Array<number>(max + 1).fill(0);
    ranges.forEach(([from, to]) => { let cursor = left; for (let r = from; r < to; r++) { x[r] = cursor + widths[r] / 2; cursor += widths[r]; } });
    const top = ranges.map(() => -Infinity), bottom = ranges.map(() => Infinity);
    graph.nodes.forEach(node => {
      const b = band[bandOf(node.id)], row = rowOf[node.id];
      top[b] = Math.max(top[b], axes[row] + dims[node.id][1] / 2);
      bottom[b] = Math.min(bottom[b], axes[row] - half[row] - LABEL_GAP - LABEL_ROOM);
    });
    // Shortcuts within a band arch over it; one that crosses to the next band runs along the corridor.
    const arched = ranges.map((_, index) => skips.some(edge => band[bandOf(edge.source)] === index && band[bandOf(edge.target)] === index));
    const roof = top.map((value, index) => value + (arched[index] ? ARC_ROOM : 0.3));
    const lift = Array<number>(ranges.length).fill(0);
    for (let b = ranges.length - 2; b >= 0; b--) lift[b] = lift[b + 1] + roof[b + 1] - bottom[b] + BAND_GAP;
    const corridors = ranges.slice(1).map((_, b) => (lift[b] + bottom[b] + lift[b + 1] + roof[b + 1]) / 2);
    const positions: Record<string, THREE.Vector3> = {}, bandOfNode: Record<string, number> = {};
    graph.nodes.forEach(node => {
      const r = stageOf[node.id], b = band[bandOf(node.id)];
      positions[node.id] = new THREE.Vector3(r <= max ? x[r] : (r - max / 2) * 2.1, axes[rowOf[node.id]] + lift[b], 0);
      bandOfNode[node.id] = b;
    });
    return { positions, bandOfNode, corridors, bands: ranges.length };
  };
  // Breaks near equal widths, each at the boundary that best trades drift from the even split
  // against the edges it makes cross.
  const total = widths.reduce((a, b) => a + b, 0), sorted = [...widths].sort((a, b) => a - b), pitch = sorted[Math.floor(sorted.length / 2)];
  const reached = widths.map((_, r) => widths.slice(0, r + 1).reduce((a, b) => a + b, 0));
  const split = (count: number) => {
    const cuts: number[] = [];
    for (let k = 1; k < count; k++) {
      let best = -1, cost = Infinity;
      for (let r = (cuts.at(-1) ?? -1) + 1; r < max - (count - 1 - k); r++) {
        const here = Math.abs(reached[r] - total * k / count) / pitch + CROSSING_COST * Math.max(0, crossings[r] - 1);
        if (here < cost) { best = r; cost = here; }
      }
      if (best < 0) break;
      cuts.push(best);
    }
    const ranges: [number, number][] = [];
    let from = 0;
    cuts.forEach(r => { ranges.push([from, r + 1]); from = r + 1; });
    ranges.push([from, max + 1]);
    return ranges;
  };
  // How large the overview camera can draw a placement in the frame.
  const offset = new THREE.Vector3(...(Object.keys(attention).length ? OVERVIEW_OFFSETS.attention : OVERVIEW_OFFSETS.blocks));
  const forward = offset.clone().negate().normalize(), right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize(), up = new THREE.Vector3().crossVectors(right, forward);
  const scale = (placed: ReturnType<typeof place>) => {
    const along = new THREE.Box2();
    graph.nodes.forEach(node => {
      const [w, h, d] = dims[node.id], p = placed.positions[node.id];
      for (const corner of [[-1, -1, -1], [1, 1, 1], [-1, 1, 1], [1, -1, -1], [-1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, 1]]) {
        const point = new THREE.Vector3(p.x + corner[0] * w / 2, p.y + corner[1] * h / 2, p.z + corner[2] * d / 2);
        along.expandByPoint(new THREE.Vector2(point.dot(right), point.dot(up)));
      }
      const label = new THREE.Vector3(p.x, p.y - h / 2 - LABEL_GAP - LABEL_ROOM, p.z);
      along.expandByPoint(new THREE.Vector2(label.dot(right), label.dot(up)));
    });
    const size = along.getSize(new THREE.Vector2());
    return Math.min(FRAME_ASPECT / Math.max(size.x, 1e-6), 1 / Math.max(size.y, 1e-6));
  };
  let best = place([[0, max + 1]]);
  const single = scale(best);
  let bestScale = single;
  for (let count = 2; count <= MAX_BANDS && max + 1 >= MIN_WRAP_STAGES; count++) {
    const ranges = split(count);
    if (ranges.length < count) break;
    const placed = place(ranges), fit = scale(placed);
    if (fit > bestScale) { best = placed; bestScale = fit; }
  }
  if (bestScale < single * WRAP_GAIN) best = place([[0, max + 1]]);
  return { ...best, rowOf, pitch };
}

/** Where an edge leaves or enters a node: an attention module's own port, or the middle of a block's side. */
function port(entry: SceneNode, side: 'input' | 'output' | 'context') {
  const offset = entry.attention ? new THREE.Vector3(...entry.attention.ports[side]) : new THREE.Vector3((side === 'input' ? -1 : 1) * entry.dims[0] / 2, 0, 0);
  return entry.position.clone().add(offset);
}

/** How far an edge from one band to the next runs on past its ends before turning along the corridor. */
const SWEEP = 1.4;

function routeEdges(graph: Graph, nodes: Record<string, SceneNode>, corridors: number[]): SceneRoute[] {
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
    // An edge that wraps to the next band leaves rightward, runs back along the corridor between
    // the bands and enters the next band's first stage from the left, like a line of text.
    if (source.band !== target.band && !shortcuts.has(edge.id)) {
      const corridor = corridors[Math.min(source.band, target.band)] ?? (start.y + end.y) / 2;
      return [{ ...edge, start, end, residual: false, control1: new THREE.Vector3(start.x + SWEEP, corridor, start.z), control2: new THREE.Vector3(end.x - SWEEP, corridor, end.z) }];
    }
    if (!shortcuts.has(edge.id)) return [{ ...edge, start, end, residual: false, control1: undefined, control2: undefined }];
    // A residual shortcut lands on top of its Add and arches over every node in between.
    end.copy(target.position).add(new THREE.Vector3(0, 0.15, 0));
    const lane = lanes.get(edge.target) ?? 0; lanes.set(edge.target, lane + 1);
    // One that crosses to the next band runs along the corridor, a little above the wrapping
    // edges, and drops onto its Add from there.
    if (source.band !== target.band) {
      const corridor = (corridors[Math.min(source.band, target.band)] ?? (start.y + end.y) / 2) + 0.3 + lane * 0.3;
      return [{ ...edge, start, end, residual: true, control1: new THREE.Vector3(start.x + SWEEP, corridor, start.z), control2: new THREE.Vector3(end.x, corridor, end.z) }];
    }
    const obstacles = graph.nodes.filter(n => n.id !== edge.source && n.id !== edge.target && boxes[n.id] && nodes[n.id].band === source.band)
      .map(n => ({ min: boxes[n.id].min.toArray(), max: boxes[n.id].max.toArray() }))
      .filter(box => box.max[0] > start.x && box.min[0] < end.x);
    const path = outerResidualCurve(start.toArray(), end.toArray(), obstacles, lane);
    return [{ ...edge, start, end, residual: true, control1: new THREE.Vector3(...path.control1), control2: new THREE.Vector3(...path.control2) }];
  });
}

/** Every node's box (with room for its label underneath), every residual arch and every wrap to the next band. */
function sceneBounds(nodes: Record<string, SceneNode>, routes: SceneRoute[]) {
  const box = new THREE.Box3();
  Object.values(nodes).forEach(entry => {
    const nodeBox = new THREE.Box3().setFromCenterAndSize(entry.position, new THREE.Vector3(...entry.dims));
    // Room for the label: its baseline and a line or two of text under it.
    nodeBox.min.y = Math.min(nodeBox.min.y - 1.1, entry.labelY - 0.5);
    box.union(nodeBox);
  });
  routes.filter(route => route.control1).forEach(route => flowCurve(route).getPoints(40).forEach(point => box.expandByPoint(point)));
  return box.isEmpty() ? new THREE.Box3(new THREE.Vector3(-5, 0, -2), new THREE.Vector3(5, 5, 2)) : box;
}

export function sceneLayout(graph: Graph, analysis: Analysis, expanded: boolean): SceneLayout {
  const attention: Record<string, AttentionGeometry> = {};
  graph.nodes.forEach(node => { if (isAttention(node.op)) attention[node.id] = attentionLayout(node, expanded); });
  const dimensions: Record<string, Dims> = {};
  graph.nodes.forEach(node => { dimensions[node.id] = attention[node.id]?.dimensions as Dims | undefined ?? blockDimensions(analysis.layers[node.id]?.output, node.op); });
  const { positions, rowOf, bandOfNode, corridors, bands, pitch } = placeNodes(graph, analysis, expanded, dimensions, attention);
  const nodes: Record<string, SceneNode> = {};
  graph.nodes.forEach(node => {
    const band = bandOfNode[node.id];
    nodes[node.id] = { node, position: positions[node.id], dims: dimensions[node.id], attention: attention[node.id] ?? null, band, row: rowOf[node.id], labelY: 0, grounded: band === bands - 1 };
  });
  // The blocks of a row hang their labels on one baseline under the row's lowest block, so labels
  // line up instead of stepping with block heights. An attention module is a tall open frame
  // centred on the same axis; measured against it, the blocks' labels would hang far below them,
  // so it labels its own lower edge instead.
  const baselines = new Map<string, number>(), bottom = (entry: SceneNode) => entry.position.y - entry.dims[1] / 2 - LABEL_GAP, key = (entry: SceneNode) => `${entry.band}:${entry.row}`;
  Object.values(nodes).forEach(entry => { if (!entry.attention) baselines.set(key(entry), Math.min(baselines.get(key(entry)) ?? Infinity, bottom(entry))); });
  Object.values(nodes).forEach(entry => { entry.labelY = entry.attention ? bottom(entry) : baselines.get(key(entry))!; });
  const routes = routeEdges(graph, nodes, corridors);
  return { nodes, routes, bounds: sceneBounds(nodes, routes), dimensions, pitch, bands };
}
