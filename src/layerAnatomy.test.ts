import { describe, expect, it } from 'vitest';
import { blockDimensions } from './sceneLayout';
import { expertAnatomy, latentAnatomy, MAX_DRAWN_EXPERTS, MAX_DRAWN_HEADS, ROPE_COLOR, type Anatomy } from './layerAnatomy';
import type { Point3 } from './flowGeometry';

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

  it('shades experts by their measured load, the busiest in the full colour', () => {
    const load = [0, 5, 10, 0, 0, 0, 0, 2];
    const experts = expertAnatomy(blockDimensions([1, 6, 16], 'MoE'), '#2f9e6e', { embed: 16, experts: 8, topK: 2, shared: 1, groups: 1, positions: 6 }, load).parts.filter(part => part.key.startsWith('expert-'));
    expect(experts[2].color).toBe('#2f9e6e');
    expect(experts[0].color).not.toBe('#2f9e6e');
    expect(experts[0].color).toBe(experts[3].color);
  });
});
