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

export interface AttentionSelection {
  branch: number;
  head: number;
  queryStart: number;
  keyStart: number;
}

export interface AttentionMatrix extends TensorSnapshot {
  rowStart: number;
  columnStart: number;
}

export const ATTENTION_TENSORS = {
  q: 'Q · 实际投影', k: 'K · 共享组实际投影', v: 'V · 共享组实际投影',
  scores: '缩放分数 · QKᵀ / √d', probabilities: 'Softmax 概率',
  headOutput: 'Head 输出 · 实际 SDPA', branchOutput: '分支输出 · Wₒ 后', mergedOutput: 'Attention 输出 · 分支平均'
} as const;
export type AttentionTensor = keyof typeof ATTENTION_TENSORS;

export interface AttentionSnapshot extends AttentionSelection {
  nodeId: string;
  kvHead: number;
  numHeads: number;
  kvHeads: number;
  branches: number;
  headDim: number;
  queryLength: number;
  keyLength: number;
  scale: number;
  mask: 'none';
  dropout: 0;
  scoreSource: 'projected-qk';
  outputSource: 'scaled_dot_product_attention';
  tensors: Record<AttentionTensor, AttentionMatrix>;
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
  attentions: AttentionSnapshot[];
}

export function tensorNumber(value: number | null) {
  if (value === null || !Number.isFinite(value)) return '非有限值';
  return value === 0 ? '0' : Math.abs(value) >= 10000 || Math.abs(value) < 0.001 ? value.toExponential(3) : Number(value.toPrecision(5)).toString();
}
