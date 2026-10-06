import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { attentionLayout, HEAD_POINTS, headPoint, visualHeads } from './attentionLayout';
import { projectionPorts } from './attentionConfig';
import { exportStacks } from './attentionExport';
import { PRESETS } from './presets';

describe('attention polygons follow the model Q and KV heads', () => {
  it('keeps the single-head triangle and ignores legacy polygon counts', () => {
    const node = PRESETS.transformer().nodes[1];
    node.params.qkv_count = 64;
    const layout = attentionLayout(node);
    expect(layout.faces[0].vertices.map(v => v.point)).toEqual([HEAD_POINTS.q, HEAD_POINTS.k, HEAD_POINTS.v]);
    delete node.params.num_heads;
    expect(visualHeads(node)).toBe(1);
  });
  it.each([1, 2, 4, 8, 16])('places %i Q heads and one shared KV pair in a single polygon', heads => {
    const node = PRESETS.mqa().nodes[1]; node.params.num_heads = heads;
    const layout = attentionLayout(node), face = layout.faces[0];
    expect(layout.faces).toHaveLength(1);
    expect(face.vertices).toHaveLength(heads + 2);
    expect(face.vertices.filter(v => v.role === 'Q')).toHaveLength(heads);
    expect(face.vertices.filter(v => v.role === 'K')).toHaveLength(1);
    expect(face.vertices.filter(v => v.role === 'V')).toHaveLength(1);
    const k = face.vertices.find(v => v.role === 'K')!, v = face.vertices.find(v => v.role === 'V')!;
    face.computations.forEach((c, i) => {
      expect(face.routes.find(r => r.id === `${k.id}-score-${i}`)?.start).toEqual(k.point);
      expect(face.routes.find(r => r.id === `${k.id}-score-${i}`)?.end).toEqual(c.score);
      expect(face.routes.find(r => r.id === `${v.id}-weighted-${i}`)?.end).toEqual(c.weighted);
    });
    expect(Object.keys(layout.projectionPositions)).toHaveLength(heads + 2);
  });
  it('groups 8 Q heads into two six-sided faces with no duplicated KV vertices', () => {
    const node = PRESETS.gqa().nodes[1], layout = attentionLayout(node);
    expect(layout.faces).toHaveLength(2);
    expect(layout.faces.map(f => f.vertices.length)).toEqual([6, 6]);
    expect(new Set(layout.faces.flatMap(f => f.vertices.map(v => v.id))).size).toBe(12);
    expect(layout.faces.map(f => f.qHeads)).toEqual([[0, 1, 2, 3], [4, 5, 6, 7]]);
    expect(layout.faces[0].position[0]).toBeLessThan(layout.faces[1].position[0]);
  });
  it.each(['transformer', 'mqa', 'gqa', 'cross_attention'])('keeps every %s port attached and every route inside camera bounds', preset => {
    const node = PRESETS[preset]().nodes[1]; node.params.branches = 3;
    for (const expanded of [false, true]) {
      const layout = attentionLayout(node, expanded);
      expect(Object.keys(layout.projectionPositions).sort()).toEqual(projectionPorts(node).map(p => p.id).sort());
      for (const face of layout.faces) {
        face.vertices.forEach(v => expect(layout.projectionPositions[v.id]).toEqual(headPoint(v.point, face.position)));
        expect(layout.routes.find(r => r.id === `fan-in-${face.index}`)?.end).toEqual(headPoint(face.points.inlet, face.position));
        face.vertices.forEach(v => expect(face.routes.find(r => r.projection === v.id)?.start).toEqual(node.params.attention_type === 'cross' && v.role !== 'Q' ? face.points.context : face.points.inlet));
      }
      layout.routes.forEach(r => {
        const curve = new THREE.CubicBezierCurve3(...[r.start, r.control1, r.control2, r.end].map(p => new THREE.Vector3(...p)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]);
        curve.getPoints(40).forEach(p => p.toArray().forEach((v, axis) => expect(Math.abs(v)).toBeLessThanOrEqual(layout.dimensions[axis] / 2)));
      });
      if (node.op === 'Transformer') {
        expect(layout.routes.find(r => r.id === 'attention-residual')?.start).toEqual(layout.ports.input);
        expect(layout.routes.find(r => r.id === 'attention-residual')?.end).toEqual(layout.ports.add1);
      }
    }
  });
});

describe('image export stacks heads that share one input space', () => {
  it.each([[8, 8], [16, 16], [8, 2]])('stacks %i Q heads over %i KV groups into one stepped column', (heads, kv) => {
    const node = PRESETS.transformer().nodes[1]; node.params.num_heads = heads; node.params.kv_heads = kv;
    const graph = PRESETS.transformer(); graph.nodes[1] = node;
    const stacks = exportStacks(graph, node), tiled = attentionLayout(node, false, true), stacked = attentionLayout(node, false, true, stacks);
    expect(stacks).toEqual([Array.from({ length: kv }, (_, i) => i)]);
    expect(stacked.dimensions[0]).toBeLessThan(tiled.dimensions[0]);
    // Every face keeps its size; each one behind the front is stepped up and right.
    expect(stacked.faces.map(f => f.vertices.map(v => v.point))).toEqual(tiled.faces.map(f => f.vertices.map(v => v.point)));
    stacked.faces.slice(1).forEach((face, i) => {
      expect(face.position[0]).toBeGreaterThan(stacked.faces[i].position[0]);
      expect(face.position[1]).toBeGreaterThan(stacked.faces[i].position[1]);
      expect(face.position[2]).toBeLessThan(stacked.faces[i].position[2]);
    });
    expect(stacked.routes.filter(r => r.id.startsWith('fan-in-'))).toHaveLength(1);
    stacked.routes.forEach(r => new THREE.CubicBezierCurve3(...[r.start, r.control1, r.control2, r.end].map(p => new THREE.Vector3(...p)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]).getPoints(40).forEach(p => p.toArray().forEach((v, axis) => expect(Math.abs(v)).toBeLessThanOrEqual(stacked.dimensions[axis] / 2))));
  });
  it('keeps independently wired heads and separate branches in their own columns', () => {
    const graph = PRESETS.transformer(), node = graph.nodes[1]; node.params.num_heads = 4; node.params.kv_heads = 4; node.params.branches = 2;
    graph.edges.push({ id: 'override', source: 'layer_0', target: node.id, targetPort: 'b0:k2' });
    expect(exportStacks(graph, node)).toEqual([[0, 1, 3], [4, 5, 6, 7]]);
    const layout = attentionLayout(node, false, true, exportStacks(graph, node));
    expect(layout.routes.filter(r => r.id.startsWith('fan-in-')).map(r => r.id)).toEqual(['fan-in-0', 'fan-in-2', 'fan-in-4']);
    expect(Object.keys(layout.projectionPositions).sort()).toEqual(projectionPorts(node).map(p => p.id).sort());
  });
});
