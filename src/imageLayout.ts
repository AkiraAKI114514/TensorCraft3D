import type { Graph } from './types';
import type { Point3 } from './flowGeometry';

export interface ImageExportOptions { layout: 'compact' | 'current'; aspect: number; fontSize: number; }
export const DEFAULT_IMAGE_OPTIONS: ImageExportOptions = { layout: 'compact', aspect: 16 / 9, fontSize: 11 };
export interface ImageBand { left: number; right: number; top: number; bottom: number; }
export interface CompactLayout { positions: Record<string, Point3>; bands: ImageBand[]; bandByNode: Record<string, number>; }

// Keep each topological stage intact, so parallel inputs stay next to each other.
// Every band flows left-to-right; wrapping never reverses the module ports.
export function compactImageLayout(graph: Graph, order: string[], dimensions: Record<string, Point3>, aspect = 16 / 9): CompactLayout {
  const rank: Record<string, number> = {}, stages: string[][] = [];
  for (const id of [...order, ...graph.nodes.map(n => n.id).filter(id => !order.includes(id))]) {
    const parents = graph.edges.filter(e => e.target === id).map(e => e.source);
    rank[id] = Math.max(-1, ...parents.map(parent => rank[parent] ?? -1)) + 1;
    (stages[rank[id]] ??= []).push(id);
  }
  const lane: Record<string, number> = {};
  stages.forEach(stage => {
    const barycenter = (id: string) => {
      const parents = graph.edges.filter(e => e.target === id).map(e => lane[e.source]).filter(v => v !== undefined);
      return parents.length ? parents.reduce((a, b) => a + b, 0) / parents.length : graph.nodes.findIndex(n => n.id === id);
    };
    stage.sort((a, b) => barycenter(a) - barycenter(b) || a.localeCompare(b));
    stage.forEach((id, index) => { lane[id] = index; });
  });
  if (!stages.length) return { positions: {}, bands: [], bandByNode: {} };
  const widths = stages.map(stage => Math.max(3.8, ...stage.map(id => dimensions[id][0] + 1.8)));
  const measure = (from: number, to: number) => {
    const ids = stages.slice(from, to).flat(), lanes = Math.max(...stages.slice(from, to).map(stage => stage.length));
    const pitch = Math.max(3.4, ...ids.map(id => dimensions[id][1] + 2));
    return { width: widths.slice(from, to).reduce((a, b) => a + b, 0), height: lanes * pitch + 1.8, pitch, lanes };
  };
  const total = widths.reduce((a, b) => a + b, 0), minimum = Math.max(...widths);
  let best: { ranges: [number, number][]; score: number } | undefined;
  // Try a range of row widths rather than a fixed number of layers: attention
  // modules need substantially more room than activations or shape operations.
  for (let trial = 0; trial <= 60; trial++) {
    const limit = minimum + (total - minimum) * trial / 60, ranges: [number, number][] = [];
    let from = 0, width = 0;
    stages.forEach((_, index) => {
      if (index > from && width + widths[index] > limit) { ranges.push([from, index]); from = index; width = 0; }
      width += widths[index];
    });
    ranges.push([from, stages.length]);
    const sizes = ranges.map(([from, to]) => measure(from, to));
    const w = Math.max(...sizes.map(s => s.width)), h = sizes.reduce((sum, s) => sum + s.height, 0);
    const cuts = graph.edges.filter(e => ranges.some(([from, to]) => rank[e.source] < to && rank[e.target] >= to && to < stages.length)).length;
    const score = Math.abs(Math.log((w / h) / aspect)) + ranges.length * 0.035 + cuts * 0.008;
    if (!best || score < best.score) best = { ranges, score };
  }
  const positions: Record<string, Point3> = {}, bandByNode: Record<string, number> = {}, bands: ImageBand[] = [];
  let top = 0;
  best!.ranges.forEach(([from, to], band) => {
    const size = measure(from, to), left = -size.width / 2;
    let x = left;
    for (let index = from; index < to; index++) {
      stages[index].forEach((id, row) => {
        positions[id] = [x + widths[index] / 2, top - 1 - size.pitch * (row + 0.5), 0];
        bandByNode[id] = band;
      });
      x += widths[index];
    }
    bands.push({ left, right: left + size.width, top, bottom: top - size.height });
    top -= size.height;
  });
  return { positions, bands, bandByNode };
}

export interface Rect { x: number; y: number; width: number; height: number; }
export const overlaps = (a: Rect, b: Rect, gap = 3) => a.x < b.x + b.width + gap && a.x + a.width + gap > b.x && a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;
export interface LabelRequest { id: string; text: string; x: number; y: number; fontSize: number; maxWidth: number; secondary?: string; color: string; }
export interface ImageLabel extends Rect { id: string; text: string; lines: string[]; fontSize: number; secondary?: string; color: string; anchor: [number, number]; }

export function wrapImageText(text: string, width: number, measure: (text: string) => number, maxLines = 3) {
  const lines: string[] = []; let line = '';
  for (const char of text) {
    if (line && measure(line + char) > width) { lines.push(line); line = ''; }
    line += char;
  }
  if (line) lines.push(line);
  if (lines.length <= maxLines) return lines;
  let last = lines[maxLines - 1];
  while (last && measure(last + '…') > width) last = last.slice(0, -1);
  return [...lines.slice(0, maxLines - 1), last + '…'];
}

// Screen-space packing is shared by PNG and SVG. Labels avoid model geometry
// and one another; bounded search prevents pathological graphs hanging export.
export function placeImageLabels(requests: LabelRequest[], obstacles: Rect[], width: number, height: number, measure: (text: string, size: number) => number): ImageLabel[] {
  const result: ImageLabel[] = [], padding = width / 1920 * 20;
  for (const request of requests) {
    const lines = wrapImageText(request.text, request.maxWidth, text => measure(text, request.fontSize));
    const w = Math.max(...lines.map(text => measure(text, request.fontSize)), request.secondary ? measure(request.secondary, request.fontSize * 0.85) : 0) + 10;
    const h = (lines.length + (request.secondary ? 1 : 0)) * request.fontSize * 1.35 + 8;
    const clamp = (x: number, y: number): Rect => ({ x: Math.max(padding, Math.min(width - padding - w, x - w / 2)), y: Math.max(padding * 2.6, Math.min(height - padding * 2 - h, y)), width: w, height: h });
    let chosen = clamp(request.x, request.y), bestPenalty = Infinity;
    for (let step = 0; step < 180; step++) {
      const ring = Math.ceil(step / 8), angle = (step % 8) * Math.PI / 4;
      const candidate = clamp(request.x + Math.cos(angle) * ring * (w / 2 + 6), request.y + Math.sin(angle) * ring * (h + 4));
      const collisions = obstacles.filter(box => overlaps(candidate, box)).length + result.filter(box => overlaps(candidate, box)).length * 4;
      if (!collisions) { chosen = candidate; break; }
      const penalty = collisions * 1000 + ring;
      if (penalty < bestPenalty) { bestPenalty = penalty; chosen = candidate; }
    }
    result.push({ ...chosen, ...request, x: chosen.x, y: chosen.y, width: chosen.width, height: chosen.height, lines, anchor: [request.x, request.y] });
  }
  return result;
}
