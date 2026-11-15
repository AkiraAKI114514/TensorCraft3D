import { describe, expect, it } from 'vitest';
import { blockDimensions } from './sceneLayout';
import { ANATOMY_OPS, blockAxis, contractionAnatomy, distributionAnatomy, DROPOUT_GRID, dropoutAnatomy, expertAnatomy, latentAnatomy, maskAnatomy, MAX_BINS, MAX_DRAWN_EXPERTS, MAX_DRAWN_HEADS, MAX_DROPOUT_CHANNELS, MAX_MASK_STEPS, ROPE_COLOR, scaleAnatomy, type Anatomy } from './layerAnatomy';
import { OPS } from './types';
import type { Point3 } from './flowGeometry';

// Every link starts and ends on a part: on its face, or inside its box.
const onParts = (anatomy: Anatomy) => anatomy.links.forEach(([from, to], index) => [from, to].forEach(point =>
  expect(anatomy.parts.some(part => part.center.every((centre, axis) => Math.abs(point[axis] - centre) <= part.size[axis] / 2 + 1e-9)), `link ${index} at ${point.map(value => value.toFixed(3))}`).toBe(true)));

const inside = (anatomy: Anatomy, box: Point3) => {
  for (const part of anatomy.parts) for (let axis = 0; axis < 3; axis++) {
    expect(Math.abs(part.center[axis]) + part.size[axis] / 2, `${part.key} axis ${axis}`).toBeLessThanOrEqual(box[axis] / 2 + 1e-9);
  }
  for (const [from, to] of anatomy.links) for (const point of [from, to]) for (let axis = 0; axis < 3; axis++) expect(Math.abs(point[axis])).toBeLessThanOrEqual(box[axis] / 2 + 1e-9);
};

// The template's MLA (src/presets.ts) and DeepSeek-V3's own (config_671B: 128 heads, q 1536, kv 512, 128 + 64 per head).
const templateLatent = { embed: 128, heads: 4, qLora: 64, kvLora: 32, nope: 16, rope: 8, value: 16, positions: 16 };
const fullLatent = { embed: 7168, heads: 128, qLora: 1536, kvLora: 512, nope: 128, rope: 64, value: 128, positions: 16 };

describe('latent attention anatomy', () => {
  it.each([['template', templateLatent], ['DeepSeek-V3', fullLatent]] as const)('fits the %s layer inside its box', (_, shape) => {
    const box = blockDimensions([1, 16, shape.embed], 'LatentAttention');
    inside(latentAnatomy(box, '#5470b4', shape), box);
  });

  it('draws both latents, the shared rotary key, a RoPE part per head and one score matrix per head', () => {
    const anatomy = latentAnatomy(blockDimensions([1, 16, 7168], 'LatentAttention'), '#5470b4', fullLatent), keys = anatomy.parts.map(part => part.key);
    expect(keys).toEqual(expect.arrayContaining(['input', 'q-latent', 'q-norm', 'kv-latent', 'kv-norm', 'k-pe', 'output']));
    expect(keys.filter(key => /^head-\d/.test(key))).toHaveLength(MAX_DRAWN_HEADS);
    expect(keys.filter(key => key.startsWith('head-rope-'))).toHaveLength(MAX_DRAWN_HEADS);
    expect(keys.filter(key => key.startsWith('scores-'))).toHaveLength(MAX_DRAWN_HEADS);
    expect(anatomy.parts.find(part => part.key === 'k-pe')!.color).toBe(ROPE_COLOR);
    // The label keeps the true head count; only the output carries measured values.
    expect(anatomy.labels.map(label => label.text)).toContain('128 heads');
    expect(anatomy.parts.filter(part => part.output).map(part => part.key)).toEqual(['output']);
    // The K/V latent is the narrower squeeze (512 vs 1536 of 7168).
    const height = (key: string) => anatomy.parts.find(part => part.key === key)!.size[1];
    expect(height('kv-latent')).toBeLessThan(height('q-latent'));
  });

  it('sets the heads on a ring around the rotary key they share, the corners of a polygon facing the flow', () => {
    const anatomy = latentAnatomy(blockDimensions([1, 16, 7168], 'LatentAttention'), '#5470b4', fullLatent);
    const hub = anatomy.parts.find(part => part.key === 'k-pe')!;
    const heads = anatomy.parts.filter(part => /^head-\d/.test(part.key));
    // A head's centre is its tile's centre; the tile spans its unrotated and RoPE parts.
    const centre = (index: number) => {
      const nope = heads[index], rope = anatomy.parts.find(part => part.key === `head-rope-${index}`)!;
      return [(nope.center[1] - nope.size[1] / 2 + rope.center[1] + rope.size[1] / 2) / 2, nope.center[2]];
    };
    const distances = heads.map((_, index) => Math.hypot(centre(index)[0] - hub.center[1], centre(index)[1] - hub.center[2]));
    distances.forEach(distance => expect(distance).toBeCloseTo(distances[0], 6));
    expect(heads.every(head => head.center[0] === hub.center[0])).toBe(true);
    const [ring, scores] = anatomy.polygons!;
    expect([ring.key, ring.corners.length, scores.key, scores.corners.length]).toEqual(['heads', MAX_DRAWN_HEADS, 'scores', MAX_DRAWN_HEADS]);
    // A spoke from the hub to every head.
    expect(anatomy.links.filter(([from]) => from[0] === hub.center[0] && from[1] === 0 && from[2] === 0)).toHaveLength(MAX_DRAWN_HEADS);
  });

  it('leaves out the query latent when queries come from one projection', () => {
    const keys = latentAnatomy(blockDimensions([1, 16, 128], 'LatentAttention'), '#5470b4', { ...templateLatent, qLora: 0 }).parts.map(part => part.key);
    expect(keys).not.toContain('q-latent');
    expect(keys).toContain('kv-latent');
  });

  it.each([['template', templateLatent], ['DeepSeek-V3', fullLatent], ['one head', { ...templateLatent, heads: 1 }], ['one query projection', { ...templateLatent, qLora: 0 }]] as const)('links only parts in the %s layer: every head reads both latents, every score matrix reaches Wᵒ', (_, shape) => {
    const anatomy = latentAnatomy(blockDimensions([1, 16, shape.embed], 'LatentAttention'), '#5470b4', shape);
    onParts(anatomy);
    const drawn = Math.min(MAX_DRAWN_HEADS, shape.heads), output = anatomy.parts.find(part => part.key === 'output')!;
    expect(anatomy.links.filter(([, to]) => to[0] === output.center[0] - output.size[0] / 2)).toHaveLength(drawn);
  });
});

describe('mixture-of-experts anatomy', () => {
  it.each([[8, 1, 1], [64, 4, 2], [256, 8, 1]])('fits %i experts in %i groups with %i shared inside its box', (experts, groups, shared) => {
    const box = blockDimensions([1, 16, 128], 'MoE');
    inside(expertAnatomy(box, '#2f9e6e', { embed: 128, experts, topK: 2, shared, groups, positions: 16 }), box);
  });

  it('draws at most 64 expert cubes, one row per group, and says what a cube stands for', () => {
    const anatomy = expertAnatomy(blockDimensions([1, 16, 7168], 'MoE'), '#2f9e6e', { embed: 7168, experts: 256, topK: 8, shared: 1, groups: 8, positions: 16 });
    const experts = anatomy.parts.filter(part => part.key.startsWith('expert-'));
    expect(experts).toHaveLength(MAX_DRAWN_EXPERTS);
    expect(new Set(experts.map(part => part.center[1].toFixed(6))).size).toBe(8);
    expect(anatomy.labels.map(label => label.text)).toEqual(expect.arrayContaining(['router · top-8', '256 experts · 8 groups (1 cube = 4)', 'shared ×1']));
  });

  it.each([[8, 1, 1], [7, 1, 0], [64, 4, 2], [256, 8, 1], [64, 64, 1], [16, 2, 1]])('links only parts with %i experts in %i groups and %i shared: the router reaches every row at a cube', (experts, groups, shared) => {
    const anatomy = expertAnatomy(blockDimensions([1, 16, 7168], 'MoE'), '#2f9e6e', { embed: 7168, experts, topK: 2, shared, groups, positions: 16 });
    onParts(anatomy);
    const router = anatomy.parts.find(part => part.key === 'router')!, rows = new Set(anatomy.parts.filter(part => part.key.startsWith('expert-')).map(part => part.center[1].toFixed(6)));
    const fanned = anatomy.links.filter(([from]) => from[0] === router.center[0] + router.size[0] / 2);
    expect(fanned).toHaveLength(Math.min(8, rows.size));
  });

  it('shades experts by their measured load, the busiest in the full colour', () => {
    const load = [0, 5, 10, 0, 0, 0, 0, 2];
    const experts = expertAnatomy(blockDimensions([1, 6, 16], 'MoE'), '#2f9e6e', { embed: 16, experts: 8, topK: 2, shared: 1, groups: 1, positions: 6 }, load).parts.filter(part => part.key.startsWith('expert-'));
    expect(experts[2].color).toBe('#2f9e6e');
    expect(experts[0].color).not.toBe('#2f9e6e');
    expect(experts[0].color).toBe(experts[3].color);
  });
});

// The smaller operators keep the box the layout gives them: every part inside it, and the parts
// together reaching all six faces, so an exported node keeps its bounds and centre.
const reaches = (anatomy: Anatomy, box: Point3) => {
  for (let axis = 0; axis < 3; axis++) {
    const low = Math.min(...anatomy.parts.map(part => part.center[axis] - part.size[axis] / 2));
    const high = Math.max(...anatomy.parts.map(part => part.center[axis] + part.size[axis] / 2));
    expect(low, `axis ${axis} low`).toBeCloseTo(-box[axis] / 2, 9);
    expect(high, `axis ${axis} high`).toBeCloseTo(box[axis] / 2, 9);
  }
};
const fits = (anatomy: Anatomy, box: Point3) => {
  inside(anatomy, box); reaches(anatomy, box); onParts(anatomy);
  expect(anatomy.parts.filter(part => part.output)).toHaveLength(1);
};

describe('operators drawn as what they do', () => {
  it('names only real operators', () => {
    for (const op of ANATOMY_OPS) expect(OPS).toContain(op);
  });

  it('maps tensor axes to block axes as tensorCells does', () => {
    expect([1, 2, 3].map(dim => blockAxis(4, dim))).toEqual([0, 1, 2]);
    expect([blockAxis(3, 1), blockAxis(3, -1), blockAxis(2, -1), blockAxis(4, 0), blockAxis(5, 1)]).toEqual([2, 1, 1, null, null]);
  });

  it.each([
    ['[B, H, S, T] scores', [1, 4, 16, 16], -2, -1, [1, 2]],
    ['[B, S, H, T] scores', [1, 16, 4, 16], 1, 3, [0, 2]],
    ['[B, S, T] scores', [1, 12, 12], -2, -1, [2, 1]]
  ] as const)('draws a causal mask on %s as a staircase in its query × key plane', (_, shape, query, key, [a, b]) => {
    const box = blockDimensions([...shape], 'CausalMask'), rank = shape.length;
    const anatomy = maskAnatomy(box, '#b0607a', { positions: shape[query < 0 ? rank + query : query], query: blockAxis(rank, query), key: blockAxis(rank, key) });
    fits(anatomy, box);
    const rows = anatomy.parts.filter(part => part.key.startsWith('row-'));
    expect(rows).toHaveLength(MAX_MASK_STEPS);
    // Row i sees keys 0..i: its length along the key axis grows by one step per row.
    rows.forEach((part, i) => {
      expect(part.cells[b]).toBe(i + 1);
      expect(part.size[b]).toBeCloseTo(box[b] * (i + 1) / MAX_MASK_STEPS, 9);
      expect(part.size[a]).toBeLessThanOrEqual(box[a] / MAX_MASK_STEPS + 1e-9);
      expect(part.size[a]).toBeGreaterThan(box[a] / MAX_MASK_STEPS * 0.85);
    });
    expect(anatomy.parts.find(part => part.output)!.key).toBe('masked');
  });

  it('keeps a short mask to its own number of rows', () => {
    const box = blockDimensions([1, 4, 4], 'CausalMask');
    expect(maskAnatomy(box, '#b0607a', { positions: 4, query: 2, key: 1 }).parts.filter(part => part.key.startsWith('row-'))).toHaveLength(4);
  });

  it.each([['class probabilities', [1, 10], -1], ['attention probabilities', [1, 4, 16, 16], -1], ['a sequence', [1, 16, 64], -1]] as const)('draws softmax over %s as a peaked distribution', (_, shape, dim) => {
    const box = blockDimensions([...shape], 'Softmax'), rank = shape.length, size = shape[rank + dim];
    for (const log of [false, true]) {
      const anatomy = distributionAnatomy(box, '#e8b84b', { bins: size, axis: blockAxis(rank, dim), log });
      fits(anatomy, box);
      const bins = anatomy.parts.filter(part => part.key.startsWith('bin-')), length = (blockAxis(rank, dim) ?? 1) === 1 ? 2 : 1;
      expect(bins).toHaveLength(Math.min(size, MAX_BINS));
      const lengths = bins.map(part => part.size[length]), longest = lengths.indexOf(Math.max(...lengths)), shortest = lengths.indexOf(Math.min(...lengths));
      // Softmax: the most likely value stands tallest. LogSoftmax: the valley's floor is there (−log p nearest 0).
      if (log) expect(shortest).toBe(Math.round((bins.length - 1) * 0.42)); else expect(longest).toBe(Math.round((bins.length - 1) * 0.42));
    }
  });

  it.each([[0.25, 'smaller'], [4, 'larger']] as const)('draws a scale by %s as plates growing %s toward the output', (factor, way) => {
    const box = blockDimensions([1, 16, 64], 'Scale'), anatomy = scaleAnatomy(box, '#879a3f', factor, [1, 16, 16]);
    fits(anatomy, box);
    const [input, , output] = anatomy.parts;
    expect(output.output).toBe(true);
    if (way === 'smaller') expect(output.size[1]).toBeLessThan(input.size[1]); else expect(output.size[1]).toBeGreaterThan(input.size[1]);
    expect(anatomy.labels[0].text).toBe(`×${factor}`);
  });

  it('draws an einsum as its operands meeting at a contraction, then the output', () => {
    const output = [1, 16, 4, 16], box = blockDimensions(output, 'Einsum');
    const anatomy = contractionAnatomy(box, '#7a6bb5', { equation: 'bshd,bthd->bsht', operands: [[1, 16, 4, 16], [1, 16, 4, 16]], output });
    fits(anatomy, box);
    expect(anatomy.parts.map(part => part.key)).toEqual(['operand-0', 'operand-1', 'contract', 'output']);
    expect(anatomy.links).toHaveLength(3);
    expect(anatomy.labels[0].text).toBe('bshd,bthd->bsht');
  });

  it('drops exactly a share p of a dropout\'s tiles, the same ones every time', () => {
    const box = blockDimensions([1, 64], 'Dropout'), count = DROPOUT_GRID * DROPOUT_GRID;
    const element = dropoutAnatomy(box, '#bc8cc8', { p: 0.3, mode: 'element', channels: 64, channelAxis: 1, seed: 'layer_3' });
    fits(element, box);
    const kept = element.parts.filter(part => part.key.startsWith('kept-')).map(part => part.key);
    expect(kept).toHaveLength(count - Math.round(count * 0.3));
    expect(dropoutAnatomy(box, '#bc8cc8', { p: 0.3, mode: 'element', channels: 64, channelAxis: 1, seed: 'layer_3' }).parts.map(part => part.key)).toEqual(element.parts.map(part => part.key));
    expect(dropoutAnatomy(box, '#bc8cc8', { p: 0.3, mode: 'element', channels: 64, channelAxis: 1, seed: 'other' }).parts.map(part => part.key)).not.toEqual(element.parts.map(part => part.key));
    // Alpha dropout keeps every tile; the dropped ones sink and turn grey.
    const alpha = dropoutAnatomy(box, '#bc8cc8', { p: 0.3, mode: 'alpha', channels: 64, channelAxis: 1, seed: 'layer_3' });
    fits(alpha, box);
    expect(alpha.parts.filter(part => part.key.startsWith('alpha-'))).toHaveLength(Math.round(count * 0.3));
    expect(alpha.parts.filter(part => part.key.startsWith('kept-') || part.key.startsWith('alpha-'))).toHaveLength(count);
  });

  it.each([['Dropout2d', [1, 16, 32, 32]], ['Dropout1d', [1, 8, 32]], ['Dropout3d', [1, 4, 8, 16, 16]]] as const)('draws %s as whole channels knocked down', (op, shape) => {
    const box = blockDimensions([...shape], op), rank = shape.length;
    const anatomy = dropoutAnatomy(box, '#bc8cc8', { p: 0.3, mode: 'channel', channels: shape[1], channelAxis: rank >= 4 ? 0 : blockAxis(rank, 1), seed: 'drop' });
    fits(anatomy, box);
    const k = Math.min(shape[1], MAX_DROPOUT_CHANNELS);
    expect(anatomy.parts).toHaveLength(k);
    expect(anatomy.parts.filter(part => part.key.startsWith('dropped-'))).toHaveLength(Math.max(1, Math.round(k * 0.3)));
  });
});
