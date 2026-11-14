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
    const rows = new Map<string, number[]>();
    for (const entry of Object.values(layout.nodes)) {
      expect(entry.labelY).toBeLessThanOrEqual(entry.position.y - entry.dims[1] / 2 - LABEL_GAP + 1e-9);
      expect(layout.bounds.min.y).toBeLessThan(entry.labelY);
      // An attention module labels its own lower edge.
      if (entry.attention) expect(entry.labelY).toBeCloseTo(entry.position.y - entry.dims[1] / 2 - LABEL_GAP);
      else rows.set(`${entry.band}:${entry.row}`, [...(rows.get(`${entry.band}:${entry.row}`) ?? []), entry.labelY]);
    }
    for (const baselines of rows.values()) expect(new Set(baselines).size).toBe(1);
  });

  it.each(['cnn', 'transformer', 'gqa', 'cross_attention', 'deepseek'])('centres every node of a row of %s on one axis, so edges along it run straight', name => {
    const graph = PRESETS[name](), layout = sceneLayout(graph, analyze(graph), false);
    const axes = new Map<string, Set<number>>(), line = (id: string) => `${layout.nodes[id].band}:${layout.nodes[id].row}`;
    for (const id of Object.keys(layout.nodes)) axes.set(line(id), new Set([...(axes.get(line(id)) ?? []), layout.nodes[id].position.y]));
    for (const heights of axes.values()) expect(heights.size).toBe(1);
    // Attention modules keep their clearance above the ground.
    for (const entry of Object.values(layout.nodes)) if (entry.attention) expect(entry.position.y - entry.dims[1] / 2).toBeGreaterThanOrEqual(1.05 - 1e-9);
    const straight = layout.routes.filter(route => !route.residual && !route.sourcePort && !route.targetPort && line(route.source) === line(route.target));
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

  it.each(['cnn', 'mlp', 'residual', 'transformer', 'mqa', 'gqa', 'cross_attention', 'decoder'])('keeps %s on one line', name => {
    const graph = PRESETS[name](), layout = sceneLayout(graph, analyze(graph), false);
    expect(layout.bands).toBe(1);
    for (const entry of Object.values(layout.nodes)) expect([entry.band, entry.grounded]).toEqual([0, true]);
  });

  it('wraps a long model into bands that read like lines of text', () => {
    const graph = PRESETS.deepseek(), layout = sceneLayout(graph, analyze(graph), false);
    expect(layout.bands).toBeGreaterThan(1);
    const entries = Object.values(layout.nodes), inBand = (band: number) => entries.filter(entry => entry.band === band);
    for (let band = 0; band < layout.bands; band++) {
      expect(inBand(band).length).toBeGreaterThan(0);
      // Only the lowest band stands on the ground.
      for (const entry of inBand(band)) expect(entry.grounded).toBe(band === layout.bands - 1);
      if (band === 0) continue;
      // Each band sits wholly below the one before it, its arches and blocks under the other's labels.
      const above = Math.min(...inBand(band - 1).map(entry => entry.labelY)), top = Math.max(...inBand(band).map(entry => entry.position.y + entry.dims[1] / 2));
      expect(top).toBeLessThan(above);
      for (const route of layout.routes.filter(route => route.residual && layout.nodes[route.source].band === band && layout.nodes[route.target].band === band)) for (const point of flowCurve(route).getPoints(20)) expect(point.y).toBeLessThan(above);
    }
    // Every band starts at the same left edge and runs left to right.
    const lefts = Array.from({ length: layout.bands }, (_, band) => Math.min(...inBand(band).map(entry => entry.position.x - entry.dims[0] / 2)));
    lefts.forEach(left => expect(left).toBeCloseTo(lefts[0], 0));
    let wrapped = 0;
    for (const route of layout.routes) {
      const source = layout.nodes[route.source], target = layout.nodes[route.target];
      if (source.band === target.band) { expect(target.position.x).toBeGreaterThan(source.position.x); continue; }
      // An edge across a break goes on to the next band, leaving rightward and running back along
      // the corridor between the bands, above the next band's blocks and below this band's labels.
      wrapped++;
      expect(target.band).toBe(source.band + 1);
      expect(route.control1!.x).toBeGreaterThan(route.start.x);
      expect(route.control1!.y).toBeCloseTo(route.control2!.y);
      expect(route.control1!.y).toBeLessThan(Math.min(...inBand(source.band).map(entry => entry.labelY)));
      expect(route.control1!.y).toBeGreaterThan(Math.max(...inBand(target.band).map(entry => entry.position.y + entry.dims[1] / 2)));
      // A shortcut drops onto its Add from the corridor; the main path enters the next band from the left.
      if (route.residual) expect(route.control2!.x).toBeCloseTo(route.end.x);
      else expect(route.control2!.x).toBeLessThan(route.end.x);
      flowCurve(route).getPoints(20).forEach(point => expect(layout.bounds.containsPoint(point)).toBe(true));
    }
    expect(wrapped).toBeGreaterThan(0);
  });

  it('splits a long model into bands of similar length', () => {
    const graph = PRESETS.deepseek(), layout = sceneLayout(graph, analyze(graph), false);
    const lengths = Array.from({ length: layout.bands }, (_, band) => {
      const xs = Object.values(layout.nodes).filter(entry => entry.band === band).map(entry => entry.position.x);
      return Math.max(...xs) - Math.min(...xs);
    });
    expect(Math.min(...lengths) / Math.max(...lengths)).toBeGreaterThan(0.6);
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
