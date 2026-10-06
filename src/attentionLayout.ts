import type { Layer } from './types';
import { residualCurve, type Point3, type FlowCurve } from './flowGeometry';
import { attentionConfig, projectionPorts, type ProjectionPort } from './attentionConfig';
export { isAttention } from './attentionConfig';
export type { Point3, FlowCurve } from './flowGeometry';
export interface AttentionRoute extends FlowCurve { id: string; color: string; head?: number; projection?: string; }
export interface QkvVertex extends ProjectionPort { point: Point3; color: string; }
export const visualHeads = (node: Layer) => Math.max(1, Math.min(16, Math.floor(Number(node.params.num_heads)) || 1));
export const visualBranches = (node: Layer) => Math.max(1, Math.min(8, Math.floor(Number(node.params.branches)) || 1));
export const visualHeadCount = (node: Layer) => visualHeads(node) * visualBranches(node);
export const HEAD_FACING: Point3 = [0, Math.PI / 2, 0];
export const headPoint = (point: Point3, center: Point3 = [0, 0, 0], flat = false): Point3 => flat ? [center[0] + point[0], center[1] + point[1], center[2] + point[2]] : [center[0] + point[2], center[1] + point[1], center[2] - point[0]];
export const HEAD_POINTS = {
  inlet: [0, 1.25, 0], q: [-0.72, 0.55, 0], k: [0.72, 0.55, 0], v: [0, -0.72, 0],
  score: [0, 0.22, 0.18], weighted: [0.65, -0.4, 0.18], outlet: [0, -1.25, 0]
} satisfies Record<string, Point3>;
export const qkvRadius = (count: number) => count === 3 ? 0.85 : Math.max(1.1, count * 0.14);
export function headPoints(count: number) {
  const portY = Math.max(1.25, qkvRadius(count) + 0.55);
  return { ...HEAD_POINTS, inlet: [0, portY, 0] as Point3, context: [0.65, portY, 0] as Point3, outlet: [0, -portY, 0] as Point3 };
}
const colors = { Q: '#319cac', K: '#d6a13d', V: '#a17cbb' };
const curve = (id: string, start: Point3, end: Point3, color: string, control1?: Point3, control2?: Point3): AttentionRoute => ({
  id, start, end, color,
  control1: control1 ?? start.map((v, i) => v + (end[i] - v) / 3) as Point3,
  control2: control2 ?? start.map((v, i) => v + (end[i] - v) * 2 / 3) as Point3
});

// One face per KV group: all of its Q heads share exactly these K and V vertices.
// `flat` turns each face toward a straight-on viewer. Export-only stacks share
// a column without changing the face size or the interactive layout.
export function attentionLayout(node: Layer, expanded = false, flat = false, stacks: number[][] = []) {
  const cfg = attentionConfig(node.params), perBranch = visualHeads(node), branches = visualBranches(node);
  const kvHeads = Math.max(1, Math.min(perBranch, Math.floor(cfg.kvHeads) || 1));
  const groups = Array.from({ length: kvHeads }, (_, group) => Array.from({ length: perBranch }, (_, head) => head).filter(head => Math.min(kvHeads - 1, Math.floor(head * kvHeads / perBranch)) === group));
  const maxSides = Math.max(...groups.map(q => q.length + 2)), radius = qkvRadius(maxSides), faceWidth = flat ? radius + 0.45 : 0, spacing = flat ? faceWidth * 2 + 0.25 : expanded ? 1.5 : 0.8;
  // A stack keeps every face at full size; each face behind the front one is
  // shifted up and right by `step`, so only a sliver of it stays visible.
  const stackOf: Record<number, number[]> = {};
  if (flat) stacks.filter(stack => stack.length > 1).forEach(stack => stack.forEach(index => { stackOf[index] = stack; }));
  const step = (n: number) => n > 1 ? Math.min(0.24, 1.8 / (n - 1)) : 0, rise = (index: number) => ((stackOf[index]?.length ?? 1) - 1) * step(stackOf[index]?.length ?? 1);
  const slots = Array.from({ length: branches }, (_, branch) => groups.map((_, group) => branch * kvHeads + group).filter(index => !stackOf[index] || stackOf[index][0] === index));
  const slotWidth = (index: number) => faceWidth * 2 + rise(index), rowWidth = (row: number[]) => row.reduce((sum, index) => sum + slotWidth(index), 0) + (row.length - 1) * 0.25;
  const maxRise = Math.max(0, ...Object.keys(stackOf).map(index => rise(Number(index))));
  const span = (kvHeads - 1) * spacing, edge = flat ? Math.max(...slots.map(rowWidth)) / 2 : span / 2 + faceWidth, branchSpacing = radius * 2 + 4.2 + maxRise, branchSpan = (branches - 1) * branchSpacing;
  const transformer = node.op === 'Transformer', preNorm = Number(node.params.norm_first ?? 1) !== 0, left = -edge - (transformer ? 2.25 : 1.65), right = edge + (transformer ? 5.35 : 1.65), offset = -(left + right) / 2;
  const fanY = radius + 1.05, ports = {
    input: [left + offset, 0, 0], norm1: [left + offset + 0.65, 0, 0], context: [left + offset, branchSpan / 2 + fanY + 0.6, 0],
    merge: [edge + offset + 1.65, 0, 0], add1: [edge + offset + 2.45, 0, 0], norm2: [edge + offset + 3.25, 0, 0],
    ffn: [edge + offset + 4.3, 0, 0], add2: [right + offset, 0, 0], output: [right + offset, 0, 0]
  } satisfies Record<string, Point3>;
  if (transformer && !preNorm) {
    ports.norm1 = [edge + offset + 3.1, 0, 0];
    ports.ffn = [edge + offset + 3.85, 0, 0];
    ports.add2 = [edge + offset + 4.65, 0, 0];
    ports.norm2 = ports.output;
  }
  const projectionPositions: Record<string, Point3> = {};
  const definitions = projectionPorts(node);
  const placed: Record<number, Point3> = {};
  if (flat) slots.forEach((row, branch) => {
    let x = offset - rowWidth(row) / 2;
    row.forEach(first => {
      const members = stackOf[first] ?? [first], d = step(members.length);
      members.forEach((index, k) => { placed[index] = [x + faceWidth + k * d, (branch - (branches - 1) / 2) * branchSpacing + (k - (members.length - 1) / 2) * d, -k * 0.06]; });
      x += slotWidth(first) + 0.25;
    });
  });
  const faces = Array.from({ length: branches * kvHeads }, (_, index) => {
    const branch = Math.floor(index / kvHeads), group = index % kvHeads, qHeads = groups[group], count = qHeads.length + 2;
    const position: Point3 = placed[index] ?? [offset + (group - (kvHeads - 1) / 2) * spacing, (branch - (branches - 1) / 2) * branchSpacing, 0];
    const vertices: QkvVertex[] = [...qHeads.map(head => definitions.find(p => p.id === `b${branch}:q${head}`)!), ...['k', 'v'].map(role => definitions.find(p => p.id === `b${branch}:${role}${group}`)!)].map((port, i) => {
      const angle = Math.PI / 2 + Math.PI / count + i * Math.PI * 2 / count;
      const point: Point3 = count === 3 ? [HEAD_POINTS.q, HEAD_POINTS.k, HEAD_POINTS.v][i] : [qkvRadius(count) * Math.cos(angle), qkvRadius(count) * Math.sin(angle), 0];
      projectionPositions[port.id] = headPoint(point, position, flat);
      return { ...port, point, color: colors[port.role] };
    });
    const points = headPoints(count), k = vertices.find(v => v.role === 'K')!, v = vertices.find(v => v.role === 'V')!;
    const computations = qHeads.map((head, i) => {
      const y = qHeads.length === 1 ? 0.22 : (i - (qHeads.length - 1) / 2) * 0.23;
      return { head: branch * perBranch + head, score: [0, y, 0.18] as Point3, weighted: [0.48, y - 0.15, 0.18] as Point3 };
    });
    const routes: AttentionRoute[] = vertices.map(vertex => ({ ...curve(`input-${vertex.id}`, cfg.type === 'cross' && vertex.role !== 'Q' ? points.context : points.inlet, vertex.point, vertex.color, [vertex.point[0], points.inlet[1], -0.25], [vertex.point[0], vertex.point[1], -0.25]), projection: vertex.id }));
    computations.forEach((c, i) => {
      const q = vertices[i];
      routes.push(curve(`${q.id}-score`, q.point, c.score, q.color), curve(`${k.id}-score-${i}`, k.point, c.score, k.color), curve(`score-weighted-${i}`, c.score, c.weighted, '#36a18a'), curve(`${v.id}-weighted-${i}`, v.point, c.weighted, v.color), curve(`weighted-output-${i}`, c.weighted, points.outlet, '#36a18a'));
    });
    return { index, branch, group, qHeads, position, vertices, points, computations, routes, stack: stackOf[index], depth: stackOf[index]?.indexOf(index) ?? 0, label: qHeads.length === 1 ? `H${qHeads[0] + 1}` : `Q${qHeads[0] + 1}-Q${qHeads.at(-1)! + 1} · KV${group + 1}` };
  });
  const headMeta = Array.from({ length: perBranch * branches }, (_, i) => ({ branch: Math.floor(i / perBranch), head: i % perBranch, kvGroup: Math.min(kvHeads - 1, Math.floor((i % perBranch) * kvHeads / perBranch)) }));
  const heads = headMeta.map(meta => faces[meta.branch * kvHeads + meta.kvGroup].position);
  const branchMerges = Array.from({ length: branches }, (_, branch): Point3 => branches === 1 ? ports.merge : [ports.merge[0] - 0.55, (branch - (branches - 1) / 2) * branchSpacing, 0]);
  const input = transformer && preNorm ? ports.norm1 : ports.input, routes: AttentionRoute[] = [];
  // A stack shares one projection of its input, so data enters and leaves
  // through the front face only.
  faces.filter(face => !face.depth).forEach(face => {
    const inlet = headPoint(face.points.inlet, face.position, flat), outlet = headPoint(face.points.outlet, face.position, flat), merge = branchMerges[face.branch];
    routes.push(curve(`fan-in-${face.index}`, input, inlet, '#728f9e', [input[0], face.position[1] + fanY, -radius - 0.3], [face.position[0], face.position[1] + fanY, -radius - 0.3]), curve(`fan-out-${face.index}`, outlet, merge, '#36a18a', [face.position[0], face.position[1] - fanY, radius + 0.3], [merge[0], face.position[1] - fanY, radius + 0.3]));
    if (cfg.type === 'cross') routes.push(curve(`context-${face.index}`, ports.context, headPoint(face.points.context, face.position, flat), '#a17cbb', [ports.context[0], face.position[1] + fanY + 0.3, radius + 0.3], [face.position[0], face.position[1] + fanY + 0.3, radius + 0.3]));
  });
  if (branches > 1) branchMerges.forEach((merge, branch) => routes.push(curve(`branch-merge-${branch}`, merge, ports.merge, '#36a18a')));
  if (transformer && preNorm) routes.push(
    curve('input-norm', ports.input, ports.norm1, '#bc8cc8'), curve('merge-add', ports.merge, ports.add1, '#36a18a'), curve('add-norm', ports.add1, ports.norm2, '#bc8cc8'), curve('norm-ffn', ports.norm2, ports.ffn, '#e48369'), curve('ffn-add', ports.ffn, ports.output, '#e48369'),
    { id: 'attention-residual', ...residualCurve(ports.input, ports.add1, branchSpan / 2 + fanY + 1, radius + 0.5) },
    { id: 'ffn-residual', ...residualCurve(ports.add1, ports.output, -branchSpan / 2 - fanY - 0.5, -radius - 0.4) }
  );
  else if (transformer) routes.push(
    curve('merge-add', ports.merge, ports.add1, '#36a18a'), curve('add-norm', ports.add1, ports.norm1, '#bc8cc8'), curve('norm-ffn', ports.norm1, ports.ffn, '#e48369'), curve('ffn-add', ports.ffn, ports.add2, '#e48369'), curve('output-norm', ports.add2, ports.norm2, '#bc8cc8'),
    { id: 'attention-residual', ...residualCurve(ports.input, ports.add1, branchSpan / 2 + fanY + 1, radius + 0.5) },
    { id: 'ffn-residual', ...residualCurve(ports.norm1, ports.add2, -branchSpan / 2 - fanY - 0.5, -radius - 0.4) }
  );
  return { count: perBranch * branches, perBranch, branches, kvHeads, faces, heads, headMeta, projectionPositions, branchMerges, qkvCount: maxSides, ports, routes, spacing, facing: flat ? [0, 0, 0] as Point3 : HEAD_FACING, dimensions: [right - left + 0.5, branchSpan + 2 * (fanY + 1.2) + maxRise, 2 * (radius + 0.75)] as Point3 };
}
