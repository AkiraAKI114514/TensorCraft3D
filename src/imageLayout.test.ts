import { describe, expect, it } from 'vitest';
import { compactImageLayout, overlaps, placeImageLabels, wrapImageText } from './imageLayout';
import { attentionLayout, isAttention, type Point3 } from './attentionLayout';
import { analyze } from './analysis';
import { imageTestGraph } from '../e2e/imageGraph';

describe('image export layout', () => {
  it.each([16 / 9, 4 / 3, 1])('wraps a dual-branch Transformer to fit aspect %s without reversing flow', aspect => {
    const graph = imageTestGraph(), analysis = analyze(graph);
    expect(analysis.valid).toBe(true);
    const dimensions = Object.fromEntries(graph.nodes.map(n => [n.id, isAttention(n.op) ? attentionLayout(n).dimensions : [0.55, 2, 0.8] as Point3]));
    const layout = compactImageLayout(graph, analysis.order, dimensions, aspect);
    expect(layout.bands.length).toBeGreaterThan(1);
    expect(layout.positions.climate[0]).toBe(layout.positions.canopy[0]);
    graph.edges.forEach(edge => {
      const a = layout.bandByNode[edge.source], b = layout.bandByNode[edge.target];
      expect(b).toBeGreaterThanOrEqual(a);
      if (a === b) expect(layout.positions[edge.target][0]).toBeGreaterThan(layout.positions[edge.source][0]);
    });
    const width = Math.max(...layout.bands.map(b => b.right - b.left)), height = -layout.bands.at(-1)!.bottom;
    expect(width / height).toBeGreaterThan(aspect * 0.65);
    expect(width / height).toBeLessThan(aspect * 1.65);
    expect(graph.nodes.every(n => layout.positions[n.id].every(Number.isFinite))).toBe(true);
  });
  it('wraps long names without squeezing the font and retains a bounded line count', () => {
    const measure = (text: string) => text.length * 6;
    const lines = wrapImageText('self.climate_projection.very_long_module_name', 70, measure);
    expect(lines).toHaveLength(3); expect(lines.at(-1)).toContain('…');
    expect(lines.every(line => measure(line) <= 70)).toBe(true);
  });
  it('places crowded captions inside the page without touching model boxes or other captions', () => {
    const requests = Array.from({ length: 12 }, (_, i) => ({ id: `label_${i}`, text: `projection_${i}`, secondary: '64 × 6 × 64', x: 400 + i * 6, y: 230, fontSize: 11, maxWidth: 110, color: '#334f5b' }));
    const obstacles = [{ x: 350, y: 150, width: 160, height: 80 }];
    const labels = placeImageLabels(requests, obstacles, 1200, 800, (text, size) => text.length * size * 0.55);
    labels.forEach((label, i) => {
      expect(label.x).toBeGreaterThanOrEqual(0); expect(label.y).toBeGreaterThanOrEqual(0);
      expect(label.x + label.width).toBeLessThanOrEqual(1200); expect(label.y + label.height).toBeLessThanOrEqual(800);
      expect(obstacles.some(box => overlaps(label, box))).toBe(false);
      expect(labels.slice(i + 1).some(other => overlaps(label, other))).toBe(false);
    });
  });
});
