import type { TrainedModelMetadata } from './trainedModel';

export interface TensorSnapshot {
  nodeId: string;
  shape: number[];
  dtype: string;
  elements: number;
  finiteCount: number;
  nonFiniteCount: number;
  stats: { min: number | null; max: number | null; mean: number | null; std: number | null };
  histogram: { edges: number[]; counts: number[] };
  slice: { indices: number[]; shape: number[]; rows: number; columns: number; values: (number | null)[][]; truncated: boolean };
}

export interface InferenceReport {
  runId: string;
  createdAt: string;
  device: 'CPU' | 'CUDA';
  seed: number;
  sampleIndex: 0;
  inputSource: 'synthetic' | 'provided';
  weights: 'random-initialized' | 'trained';
  model: TrainedModelMetadata | null;
  inputTransform: 'none' | 'csv-standardized';
  mode: 'eval';
  tensors: TensorSnapshot[];
}

export function tensorNumber(value: number | null) {
  if (value === null || !Number.isFinite(value)) return '非有限值';
  return value === 0 ? '0' : Math.abs(value) >= 10000 || Math.abs(value) < 0.001 ? value.toExponential(3) : Number(value.toPrecision(5)).toString();
}
