import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { analyze } from './analysis';
import { attentionLayout } from './attentionLayout';
import { PRESETS } from './presets';
import { blockDimensions, flowCurve, LABEL_GAP, sceneLayout } from './sceneLayout';

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

  it.each(['cnn', 'transformer', 'cross_attention', 'deepseek'])('hangs the labels of the blocks of each row of %s on one baseline under them', name => {
    const graph = PRESETS[name](), layout = sceneLayout(graph, analyze(graph), false);
    const rows = new Map<number, number[]>();
    for (const entry of Object.values(layout.nodes)) {
      expect(entry.labelY).toBeLessThanOrEqual(entry.position.y - entry.dims[1] / 2 - LABEL_GAP + 1e-9);
      expect(layout.bounds.min.y).toBeLessThan(entry.labelY);
      // An attention module labels its own lower edge.
      if (entry.attention) expect(entry.labelY).toBeCloseTo(entry.position.y - entry.dims[1] / 2 - LABEL_GAP);
      else rows.set(entry.row, [...(rows.get(entry.row) ?? []), entry.labelY]);
    }
    for (const baselines of rows.values()) expect(new Set(baselines).size).toBe(1);
  });

  it.each(['cnn', 'transformer', 'gqa', 'cross_attention', 'deepseek'])('centres every node of a row of %s on one axis, so edges along it run straight', name => {
    const graph = PRESETS[name](), layout = sceneLayout(graph, analyze(graph), false);
    const axes = new Map<number, Set<number>>();
    for (const entry of Object.values(layout.nodes)) axes.set(entry.row, new Set([...(axes.get(entry.row) ?? []), entry.position.y]));
    for (const heights of axes.values()) expect(heights.size).toBe(1);
    // Attention modules keep their clearance above the ground.
    for (const entry of Object.values(layout.nodes)) if (entry.attention) expect(entry.position.y - entry.dims[1] / 2).toBeGreaterThanOrEqual(1.05 - 1e-9);
    const straight = layout.routes.filter(route => !route.residual && !route.sourcePort && !route.targetPort && layout.nodes[route.source].row === layout.nodes[route.target].row);
    expect(straight.length).toBeGreaterThan(0);
    for (const route of straight) expect(route.end.y).toBeCloseTo(route.start.y);
  });

  it('stacks parallel rows with room for labels between them', () => {
    const graph = PRESETS.cross_attention(), layout = sceneLayout(graph, analyze(graph), false);
    const rows = [...new Set(Object.values(layout.nodes).map(entry => entry.row))].sort();
    expect(rows.length).toBeGreaterThan(1);
    for (const row of rows.slice(1)) {
      const lower = Object.values(layout.nodes).filter(entry => entry.row === row - 1), upper = Object.values(layout.nodes).filter(entry => entry.row === row);
      const top = Math.max(...lower.map(entry => entry.position.y + entry.dims[1] / 2));
      for (const entry of upper) expect(entry.labelY).toBeGreaterThan(top);
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
