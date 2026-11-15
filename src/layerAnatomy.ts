import { COLORS } from './types';
import type { Point3 } from './flowGeometry';
import { tensorCells } from './tensorBlock';

/**
 * The inside of a latent attention or mixture-of-experts layer, laid out in the layer's own box:
 * the parts its computation passes through, the links between them and the names they carry. The
 * scene draws each part as a small tensor block, so they shade, glow and export like any block.
 */
export interface AnatomyPart {
  key: string;
  center: Point3;
  size: Point3;
  color: string;
  cells: Point3;
  /** The part that stands for the layer's output, which measured values paint. */
  output?: boolean;
}
export interface AnatomyLabel { text: string; position: Point3; color: string }
/**
 * A frosted polygon facing the flow (in the y–z plane at `x`), its corners `corners` as (y, z): the
 * panel attention heads share, as on the attention module's polygon faces.
 */
export interface AnatomyPolygon { key: string; x: number; corners: [number, number][]; color: string }
export interface Anatomy { parts: AnatomyPart[]; links: [Point3, Point3][]; labels: AnatomyLabel[]; polygons?: AnatomyPolygon[] }

/** Colours of the parts that are not the layer's own: the rotary key and RMSNorm. */
export const ROPE_COLOR = '#e0a33a';
const NORM_COLOR = COLORS.RMSNorm;

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const mix = (color: string, toward: string, amount: number) => {
  const a = parseInt(color.slice(1), 16), b = parseInt(toward.slice(1), 16);
  const channel = (shift: number) => Math.round(((a >> shift) & 255) + ((((b >> shift) & 255) - ((a >> shift) & 255)) * amount));
  return `#${[16, 8, 0].map(shift => channel(shift).toString(16).padStart(2, '0')).join('')}`;
};
const tint = (color: string, amount: number) => mix(color, '#ffffff', amount);
const shade = (color: string, amount: number) => mix(color, '#000000', amount);
/** From the right face of one part to the left face of another. */
const link = (from: AnatomyPart, to: AnatomyPart, fromY = from.center[1], toY = to.center[1]): [Point3, Point3] =>
  [[from.center[0] + from.size[0] / 2, fromY, from.center[2]], [to.center[0] - to.size[0] / 2, toY, to.center[2]]];

/** At most this many heads are drawn; the label gives the true count. */
export const MAX_DRAWN_HEADS = 8;

export interface LatentShape { embed: number; heads: number; qLora: number; kvLora: number; nope: number; rope: number; value: number; positions: number }

/**
 * Multi-head latent attention, left to right:
 * - the input, then two lanes: Q is squeezed through its own latent c_Q (upper), and K and V
 *   share the latent c_KV (lower); each latent is RMS-normalised;
 * - the heads on a ring, the corners of a frosted polygon facing the flow as on the attention
 *   module's heads, each split into the part that is not rotated and the RoPE part on top; at the
 *   hub the rotary key k_pe, which every head shares, with a spoke to each head;
 * - the score matrices (positions × positions) on a second, smaller ring behind;
 * - the output, after the heads are concatenated and projected.
 * Latent heights follow the square root of their rank over the embedding width, so the squeeze shows.
 */
export function latentAnatomy([width, height, depth]: Point3, color: string, shape: LatentShape): Anatomy {
  const left = -width / 2, x = (fraction: number) => left + width * fraction, lane = height / 2;
  const S = shape.positions, latentWidth = width * 0.075, latentDepth = depth * 0.8;
  const input: AnatomyPart = { key: 'input', center: [x(0.05), 0, 0], size: [width * 0.09, height, depth], color: tint(color, 0.3), cells: tensorCells([1, S, shape.embed]) };
  const parts: AnatomyPart[] = [input], links: [Point3, Point3][] = [], labels: AnatomyLabel[] = [];
  // A square-root scale keeps a small rank visible (DeepSeek-V3 squeezes 7168 into 512) and in order.
  const latentHeight = (rank: number) => lane * 0.9 * clamp(Math.sqrt(rank / Math.max(1, shape.embed)), 0.15, 1);
  const query: AnatomyPart | null = shape.qLora ? { key: 'q-latent', center: [x(0.24), lane / 2, 0], size: [latentWidth, latentHeight(shape.qLora), latentDepth], color: tint(color, 0.12), cells: tensorCells([1, S, shape.qLora]) } : null;
  const kv: AnatomyPart = { key: 'kv-latent', center: [x(0.24), -lane / 2, 0], size: [latentWidth, latentHeight(shape.kvLora), latentDepth], color: shade(color, 0.18), cells: tensorCells([1, S, shape.kvLora]) };
  const norm = (of: AnatomyPart, key: string): AnatomyPart => ({ key, center: [of.center[0] + latentWidth / 2 + width * 0.035, of.center[1], 0], size: [width * 0.02, of.size[1], latentDepth], color: NORM_COLOR, cells: [1, 1, 1] });
  const kvNorm = norm(kv, 'kv-norm');
  let qNorm: AnatomyPart | null = null;
  if (query) {
    qNorm = norm(query, 'q-norm');
    parts.push(query, qNorm);
    links.push(link(input, query), link(query, qNorm));
    labels.push({ text: 'c_Q', position: [query.center[0], query.center[1] + query.size[1] / 2 + 0.16, 0], color });
  }
  parts.push(kv, kvNorm);
  links.push(link(input, kv), link(kv, kvNorm));
  labels.push({ text: 'c_KV', position: [kv.center[0], kv.center[1] - kv.size[1] / 2 - 0.16, 0], color });
  // The heads on a ring facing the flow, the first at the top; k_pe at the hub.
  const drawn = Math.max(1, Math.min(MAX_DRAWN_HEADS, shape.heads)), radius = Math.min(height, depth) * 0.36;
  const headX = x(0.55), headWidth = width * 0.045, ropeShare = clamp(shape.rope / Math.max(1, shape.nope + shape.rope), 0.1, 0.6);
  const tile = Math.min(0.3, drawn > 1 ? 2 * Math.PI * radius / drawn * 0.5 : 0.3), ring = (head: number, scale = 1): [number, number] => {
    const angle = Math.PI / 2 + 2 * Math.PI * head / drawn;
    return drawn > 1 ? [Math.sin(angle) * radius * scale, Math.cos(angle) * radius * scale] : [0, 0];
  };
  const hub = Math.max(0.08, radius * 0.3);
  const ropeKey: AnatomyPart = { key: 'k-pe', center: [headX, 0, 0], size: [headWidth * 0.9, hub, hub], color: ROPE_COLOR, cells: tensorCells([1, S, shape.rope]) };
  if (drawn > 1) parts.push(ropeKey);
  const scoreX = x(0.76), plate = Math.min(tile * 0.95, 0.26);
  const output: AnatomyPart = { key: 'output', center: [x(0.93), 0, 0], size: [width * 0.14, height, depth], color, cells: tensorCells([1, S, shape.embed]), output: true };
  const rightOf = (part: AnatomyPart): Point3 => [part.center[0] + part.size[0] / 2, part.center[1], part.center[2]];
  for (let head = 0; head < drawn; head++) {
    const [y, z] = ring(head), [scoreY, scoreZ] = ring(head, 0.88);
    const nopeHeight = tile * (1 - ropeShare), ropeHeight = tile * ropeShare;
    parts.push({ key: `head-${head}`, center: [headX, y - tile / 2 + nopeHeight / 2, z], size: [headWidth, nopeHeight, tile * 0.55], color: tint(color, 0.18), cells: tensorCells([1, shape.nope]) });
    parts.push({ key: `head-rope-${head}`, center: [headX, y + tile / 2 - ropeHeight / 2, z], size: [headWidth, ropeHeight, tile * 0.55], color: ROPE_COLOR, cells: tensorCells([1, shape.rope]) });
    parts.push({ key: `scores-${head}`, center: [scoreX, scoreY, scoreZ], size: [width * 0.015, plate, plate], color: shade(color, 0.3), cells: tensorCells([1, S, S]) });
    // Every head reads both latents, at the middle of its tile; a spoke from the shared rotary key
    // reaches it too. The head goes on to its score matrix, and that to Wᵒ.
    // Each leaves its latent's face toward the head, so the links run as a bundle instead of
    // meeting in one point, where at night their glow piled up into a white spot.
    const tileLeft: Point3 = [headX - headWidth / 2, y, z], toward = (latent: AnatomyPart): Point3 => {
      const [x0, y0] = rightOf(latent);
      return [x0, y0 + y / radius * latent.size[1] * 0.35, z / radius * latent.size[2] * 0.35];
    };
    if (qNorm) links.push([toward(qNorm), tileLeft]);
    links.push([toward(kvNorm), tileLeft]);
    if (drawn > 1) links.push([[headX, 0, 0], [headX, y, z]]);
    links.push([[headX + headWidth / 2, y, z], [scoreX - width * 0.0075, scoreY, scoreZ]], [[scoreX + width * 0.0075, scoreY, scoreZ], [output.center[0] - output.size[0] / 2, scoreY, scoreZ]]);
  }
  // A lone head sits at the hub, so its rotary key hangs under it.
  const rotary: AnatomyPart = drawn === 1 ? { ...ropeKey, center: [headX, -tile / 2 - hub, 0] } : ropeKey;
  if (drawn === 1) parts.push(rotary);
  parts.push(output);
  links.push(link(kvNorm, rotary, kvNorm.center[1], rotary.center[1]));
  const corners = (scale: number) => Array.from({ length: drawn }, (_, head) => ring(head, scale));
  const polygons: AnatomyPolygon[] = drawn > 2 ? [{ key: 'heads', x: headX, corners: corners(1), color }, { key: 'scores', x: scoreX, corners: corners(0.88), color: shade(color, 0.3) }] : [];
  labels.push(
    { text: `${shape.heads} heads`, position: [headX, radius + tile / 2 + 0.2, 0], color },
    { text: 'k_pe · RoPE', position: [headX, -radius - tile / 2 - 0.18, 0], color: ROPE_COLOR },
    { text: 'softmax(QKᵀ)·V', position: [scoreX, radius + plate / 2 + 0.2, 0], color },
    { text: 'Wᵒ', position: [output.center[0], height / 2 + 0.18, 0], color }
  );
  return { parts, links, labels, polygons };
}

/** At most this many expert cubes are drawn; past that, each cube stands for a run of experts. */
export const MAX_DRAWN_EXPERTS = 64;

export interface ExpertShape { embed: number; experts: number; topK: number; shared: number; groups: number; positions: number }

/**
 * A mixture-of-experts layer, left to right: the input; the router, which scores every expert and
 * keeps the top k; the bank of routed experts, one row per expert group when there are groups; the
 * shared experts, which every position goes through; the weighted sum; the output. A measured
 * routing (`load`, positions per expert) shades each expert cube from pale (none) to its full colour
 * (the busiest), so the experts a sample actually used stand out.
 */
export function expertAnatomy([width, height, depth]: Point3, color: string, shape: ExpertShape, load?: number[]): Anatomy {
  const left = -width / 2, x = (fraction: number) => left + width * fraction, S = shape.positions;
  const input: AnatomyPart = { key: 'input', center: [x(0.045), 0, 0], size: [width * 0.07, height, depth], color: tint(color, 0.3), cells: tensorCells([1, S, shape.embed]) };
  const router: AnatomyPart = { key: 'router', center: [x(0.17), 0, 0], size: [width * 0.04, height * 0.8, depth * 0.62], color: tint(color, 0.45), cells: tensorCells([1, S, shape.experts]) };
  const per = Math.ceil(Math.max(1, shape.experts) / MAX_DRAWN_EXPERTS), drawn = Math.ceil(Math.max(1, shape.experts) / per);
  const groups = Math.max(1, Math.min(shape.groups, drawn));
  const rows = groups > 1 ? groups : Math.max(1, Math.round(Math.sqrt(drawn * height / depth)));
  const columns = Math.ceil(drawn / rows);
  const bankX = x(0.43), bankHeight = height * 0.92, bankDepth = depth * 0.92;
  const cell = Math.min(bankHeight / rows, bankDepth / columns), cube = cell * 0.7;
  const loads = load ? Array.from({ length: drawn }, (_, index) => load.slice(index * per, (index + 1) * per).reduce((sum, count) => sum + count, 0)) : null;
  const busiest = loads ? Math.max(1, ...loads) : 1;
  const parts: AnatomyPart[] = [input, router];
  for (let index = 0; index < drawn; index++) {
    const row = Math.floor(index / columns), column = index % columns;
    const y = bankHeight / 2 - cell * (row + 0.5) - (bankHeight - rows * cell) / 2, z = -bankDepth / 2 + cell * (column + 0.5) + (bankDepth - columns * cell) / 2;
    parts.push({ key: `expert-${index}`, center: [bankX, y, z], size: [cube * 1.25, cube, cube], color: loads ? mix(tint(color, 0.78), color, loads[index] / busiest) : color, cells: [1, 1, 1] });
  }
  const shared = Math.max(0, Math.min(4, shape.shared)), sharedSize = Math.min(height * 0.55 / Math.max(1, shared), depth * 0.3, 0.42);
  const step = sharedSize * 1.15;
  const sharedParts = Array.from({ length: shared }, (_, index): AnatomyPart => ({ key: `shared-${index}`, center: [x(0.7), ((shared - 1) / 2 - index) * step, 0], size: [sharedSize * 0.8, sharedSize, sharedSize], color: shade(color, 0.12), cells: [1, 1, 1] }));
  parts.push(...sharedParts);
  const combine: AnatomyPart = { key: 'combine', center: [x(0.83), 0, 0], size: [0.2, 0.2, 0.2], color: shade(color, 0.12), cells: [1, 1, 1] };
  const output: AnatomyPart = { key: 'output', center: [x(0.945), 0, 0], size: [width * 0.11, height, depth], color, cells: tensorCells([1, S, shape.embed]), output: true };
  parts.push(combine, output);
  // The router reaches each row of cubes (a group, when there are groups) at its cube nearest the
  // middle, at most eight rows spread evenly, and each such cube reaches the combine. Links at fixed
  // heights on the bank's centre ended between cubes whenever the rows or columns were even, as with
  // DeepSeek-V3's eight groups of eight.
  const experts = parts.filter(part => part.key.startsWith('expert-'));
  const reached = Array.from({ length: rows }, (_, row) => experts.slice(row * columns, (row + 1) * columns))
    .filter(row => row.length).map(row => row.reduce((best, part) => Math.abs(part.center[2]) < Math.abs(best.center[2]) ? part : best));
  const fanned = reached.length <= 8 ? reached : Array.from({ length: 8 }, (_, index) => reached[Math.round(index * (reached.length - 1) / 7)]);
  const links: [Point3, Point3][] = [link(input, router), link(combine, output)];
  for (const target of fanned) links.push(link(router, target, target.center[1] * router.size[1] / bankHeight * 0.9, target.center[1]), link(target, combine, target.center[1], 0));
  for (const part of sharedParts) links.push(link(input, part, input.center[1], part.center[1]), link(part, combine));
  const labels: AnatomyLabel[] = [
    { text: `router · top-${shape.topK}`, position: [router.center[0], router.size[1] / 2 + 0.18, 0], color },
    { text: `${shape.experts} experts${shape.groups > 1 ? ` · ${shape.groups} groups` : ''}${per > 1 ? ` (1 cube = ${per})` : ''}`, position: [bankX, bankHeight / 2 + 0.2, 0], color },
    { text: 'Σ wᵢ·Eᵢ(x)', position: [combine.center[0], combine.size[1] / 2 + 0.2, 0], color }
  ];
  if (shared) labels.push({ text: `shared ×${shape.shared}`, position: [x(0.7), -(shared * step) / 2 - 0.16, 0], color });
  return { parts, links, labels };
}
