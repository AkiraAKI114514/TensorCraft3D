import type { InferenceReport, TensorSnapshot } from './inferenceTypes';
import type { ObservationSnapshot } from './observationSnapshot';
import { computationKey } from './trainedModel';
import type { Graph } from './types';

/**
 * Measured values for every layer, painted on the 3D blocks: one "all layers" run of the Tensor
 * Inspector, or an offline snapshot. Like the inspector's own results it is bound to the
 * computation it was measured on; the scene paints it only while the graph still computes that.
 */
export interface SceneObservation {
  key: string;
  graphKey: string;
  source: 'live' | 'offline';
  weights: 'trained' | 'random-initialized' | 'external';
  createdAt: string;
  tensors: Record<string, TensorSnapshot>;
}

export const observationFromReport = (report: InferenceReport, graph: Graph): SceneObservation => ({
  key: report.runId, graphKey: computationKey(graph), source: 'live', weights: report.weights, createdAt: report.createdAt,
  tensors: Object.fromEntries(report.tensors.map(tensor => [tensor.nodeId, tensor])),
});

export const observationFromSnapshot = (snapshot: ObservationSnapshot): SceneObservation => ({
  key: snapshot.snapshotId, graphKey: computationKey(snapshot.graph), source: 'offline', weights: snapshot.weights, createdAt: snapshot.createdAt,
  tensors: Object.fromEntries(snapshot.tensors.map(tensor => [tensor.nodeId, tensor])),
});

/** Stored for a non-finite element; the block draws it in a colour of its own. Float32, as stored. */
export const MISSING = Math.fround(1e30);

/** A block axis, as in tensorBlock.ts: 0 = thickness (x), 1 = height (y), 2 = depth (z). */
export type BlockAxis = 0 | 1 | 2;

export interface BlockPaint {
  /** The measured window, laid on the faces spanned by `u` (its columns) and `v` (its rows). */
  map: { values: Float32Array; columns: number; rows: number; u: BlockAxis; v: BlockAxis; uTotal: number; vTotal: number } | null;
  /** The mean of each captured channel, along `axis` on the faces that span it. */
  strip: { values: Float32Array; axis: BlockAxis; total: number; start: number; count: number } | null;
  /** Symmetric colour range: the largest magnitude in the whole tensor, as in the inspector. */
  limit: number;
}

const finite = (value: number | null) => value === null || !Number.isFinite(value) ? MISSING : value;

/**
 * Where a layer's measured tensor lands on its block. Element 0 of every axis sits at the top,
 * front corner, so a window reads like the inspector's table: a feature map's window covers its
 * large face rows down and columns across, and its channel means run through the thickness on
 * the other faces, channel 0 in front. A sequence's window puts positions across and features
 * down; a vector shows its features down. Only the window the inspector captured is painted, so
 * on a map larger than 16 × 16 it covers the top corner and the rest of the face stays unpainted.
 */
export function blockPaint(tensor: TensorSnapshot): BlockPaint | null {
  const { shape, slice, channels, stats } = tensor;
  const limit = Math.max(Math.abs(stats.min ?? 0), Math.abs(stats.max ?? 0));
  const values = Float32Array.from(slice.values.flat(), finite);
  const window = { values, columns: slice.columns, rows: slice.rows };
  if (shape.length === 4) return {
    map: { ...window, u: 2, v: 1, uTotal: shape[3], vTotal: shape[2] },
    strip: channels ? { values: Float32Array.from(channels.stats, stat => finite(stat.mean)), axis: 0, total: shape[1], start: channels.start, count: channels.count } : null,
    limit,
  };
  if (shape.length === 3) return { map: { ...window, u: 1, v: 2, uTotal: shape[2], vTotal: shape[1] }, strip: null, limit };
  if (shape.length === 2) return { map: { ...window, u: 1, v: 2, uTotal: shape[1], vTotal: 1 }, strip: null, limit };
  return null;
}
