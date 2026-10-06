import type { Graph } from './types';
import { residualCurve, type Point3 } from './flowGeometry';

export interface NodeBounds { min: Point3; max: Point3; }

export function residualEdges(graph: Graph): Set<string> {
  const successors = new Map<string, string[]>();
  graph.edges.forEach(edge => successors.set(edge.source, [...(successors.get(edge.source) ?? []), edge.target]));
  const reaches = (source: string, target: string) => {
    const pending = [...(successors.get(source) ?? [])], seen = new Set([source]);
    while (pending.length) {
      const id = pending.pop()!;
      if (id === target) return true;
      if (seen.has(id)) continue;
      seen.add(id); pending.push(...(successors.get(id) ?? []));
    }
    return false;
  };
  const residual = new Set<string>();
  graph.nodes.filter(node => node.op === 'Add').forEach(node => {
    const incoming = graph.edges.filter(edge => edge.target === node.id);
    incoming.forEach(edge => {
      if (incoming.some(other => other.id !== edge.id && reaches(edge.source, other.source))) residual.add(edge.id);
    });
  });
  return residual;
}

export function outerResidualCurve(start: Point3, end: Point3, obstacles: NodeBounds[], lane = 0) {
  const clearance = 0.55 + lane * 0.3;
  let height = Math.max(start[1], end[1]) + 2 + lane * 0.4;
  const width = end[0] - start[0];
  // Find the cubic parameter at each obstacle edge, then raise the whole arch above it.
  if (width > 0) obstacles.forEach(box => {
    for (const x of [box.min[0] - 0.12, (box.min[0] + box.max[0]) / 2, box.max[0] + 0.12]) {
      if (x <= start[0] || x >= end[0]) continue;
      let low = 0, high = 1;
      for (let i = 0; i < 30; i++) {
        const t = (low + high) / 2, px = start[0] + width * t * t * (3 - 2 * t);
        if (px < x) low = t; else high = t;
      }
      const t = (low + high) / 2, r = 1 - t;
      height = Math.max(height, (box.max[1] + clearance - r ** 3 * start[1] - t ** 3 * end[1]) / (3 * t * r));
    }
  });
  return residualCurve(start, end, height, Math.max(start[2], end[2]) + 0.6 + lane * 0.15);
}
