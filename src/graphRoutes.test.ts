import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { outerResidualCurve, residualEdges, type NodeBounds } from './graphRoutes';
import { RESIDUAL_COLOR, type FlowCurve, type Point3 } from './flowGeometry';
import { PRESETS } from './presets';
import { COLORS, type Graph } from './types';
import { attentionLayout } from './attentionLayout';

function curve(route: FlowCurve) {
  return new THREE.CubicBezierCurve3(...[route.start, route.control1, route.control2, route.end].map(p => new THREE.Vector3(...p)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]);
}

describe('residual routes', () => {
  it('distinguishes the shortcut from the transformed input at Add', () => {
    expect([...residualEdges(PRESETS.residual())]).toEqual(['skip']);
  });

  it('does not depend on edge names or graph array order', () => {
    const graph = PRESETS.residual();
    graph.edges = graph.edges.map((edge, index) => ({ ...edge, id: `connection-${index}` })).reverse();
    graph.nodes.reverse();
    expect([...residualEdges(graph)]).toEqual(['connection-10']);
  });

  it('keeps independent parallel branches as ordinary Add inputs', () => {
    const graph: Graph = { version: 1, name: 'parallel', nodes: ['input', 'left', 'right', 'add'].map(id => ({ id, name: id, op: id === 'add' ? 'Add' : id === 'input' ? 'Input' : 'ReLU', params: {}, position: { x: 0, y: 0 } })), edges: [
      { id: 'a', source: 'input', target: 'left' }, { id: 'b', source: 'input', target: 'right' },
      { id: 'c', source: 'left', target: 'add' }, { id: 'd', source: 'right', target: 'add' }
    ] };
    expect(residualEdges(graph).size).toBe(0);
  });

  it('finishes traversal when an invalid graph contains a cycle', () => {
    const graph = PRESETS.residual();
    graph.edges.push({ id: 'cycle', source: 'layer_3', target: 'layer_2' });
    expect([...residualEdges(graph)]).toEqual(['skip']);
  });

  it.each([0, 1, 2])('routes lane %i above intervening module bounds', lane => {
    const start: Point3 = [0, 1.65, 0], end: Point3 = [8, 1.8, 0];
    const obstacles: NodeBounds[] = [{ min: [1.6, 0.5, -1], max: [2.4, 3, 1] }, { min: [5.8, 0, -1.2], max: [6.4, 4.5, 1.2] }];
    const path = outerResidualCurve(start, end, obstacles, lane), samples = curve(path).getPoints(1000);
    expect(samples[0].toArray()).toEqual(start);
    expect(samples.at(-1)!.toArray()).toEqual(end);
    obstacles.forEach(box => samples.filter(p => p.x >= box.min[0] && p.x <= box.max[0]).forEach(p => expect(p.y).toBeGreaterThan(box.max[1] + 0.5)));
    expect(samples.every(p => Number.isFinite(p.x + p.y + p.z))).toBe(true);
  });

  it('shares pink color with both encoder bypasses and Add', () => {
    const routes = attentionLayout(PRESETS.transformer().nodes[1]).routes.filter(route => route.id.endsWith('residual'));
    expect(routes).toHaveLength(2);
    expect(routes.every(route => route.color === RESIDUAL_COLOR)).toBe(true);
    expect(COLORS.Add).toBe(RESIDUAL_COLOR);
    expect(outerResidualCurve([0, 0, 0], [2, 0, 0], []).color).toBe(RESIDUAL_COLOR);
  });
});
