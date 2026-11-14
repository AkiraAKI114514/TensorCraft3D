import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { analyze } from './analysis';
import { attentionLayout } from './attentionLayout';
import { PRESETS } from './presets';
import { blockDimensions, flowCurve, sceneLayout } from './sceneLayout';

const boxOf = (position: THREE.Vector3, dims: [number, number, number]) => new THREE.Box3().setFromCenterAndSize(position, new THREE.Vector3(...dims));

describe('scene layout', () => {
  it.each(Object.keys(PRESETS))('places every layer and routes every edge of %s', name => {
    const graph = PRESETS[name](), analysis = analyze(graph), layout = sceneLayout(graph, analysis, false);
    expect(Object.keys(layout.nodes).sort()).toEqual(graph.nodes.map(node => node.id).sort());
    expect(layout.routes.map(route => route.id)).toEqual(graph.edges.map(edge => edge.id));
    for (const entry of Object.values(layout.nodes)) {
      expect(layout.dimensions[entry.node.id]).toBe(entry.dims);
      // The bounds hold every layer, so the overview never crops one.
      expect(layout.bounds.containsBox(boxOf(entry.position, entry.dims))).toBe(true);
    }
  });

  it('sizes blocks from their output and attention modules from their own geometry', () => {
    const graph = PRESETS.transformer(), analysis = analyze(graph), layout = sceneLayout(graph, analysis, false);
    for (const node of graph.nodes) {
      const entry = layout.nodes[node.id];
      if (entry.attention) expect(entry.dims).toEqual(attentionLayout(node, false).dimensions);
      else expect(entry.dims).toEqual(blockDimensions(analysis.layers[node.id]?.output, node.op));
    }
  });

  it('runs a chain left to right, one stage per layer', () => {
    const graph = PRESETS.cnn(), layout = sceneLayout(graph, analyze(graph), false);
    const xs = graph.nodes.map(node => layout.nodes[node.id].position.x);
    expect(xs).toEqual([...xs].sort((a, b) => a - b));
    // Edges leave a block's right face and enter the next block's left face.
    for (const route of layout.routes) {
      const source = layout.nodes[route.source], target = layout.nodes[route.target];
      expect(route.start.x).toBeCloseTo(source.position.x + source.dims[0] / 2);
      expect(route.end.x).toBeCloseTo(target.position.x - target.dims[0] / 2);
    }
  });

  it('arches a residual shortcut over the layers it skips and keeps the arch in the bounds', () => {
    const graph = PRESETS.residual(), layout = sceneLayout(graph, analyze(graph), false);
    const shortcut = layout.routes.find(route => route.residual)!;
    expect(shortcut.id).toBe('skip');
    const points = flowCurve(shortcut).getPoints(40);
    const skipped = Object.values(layout.nodes).filter(entry => entry.node.id !== shortcut.source && entry.node.id !== shortcut.target && entry.position.x > shortcut.start.x && entry.position.x < shortcut.end.x);
    expect(skipped.length).toBeGreaterThan(0);
    for (const entry of skipped) {
      const top = entry.position.y + entry.dims[1] / 2;
      const above = points.filter(point => Math.abs(point.x - entry.position.x) < entry.dims[0] / 2);
      for (const point of above) expect(point.y).toBeGreaterThan(top);
    }
    points.forEach(point => expect(layout.bounds.containsPoint(point)).toBe(true));
  });
});
