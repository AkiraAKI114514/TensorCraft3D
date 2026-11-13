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
  /** Feature maps for the page of channels around the selected one; only `[1,C,H,W]` and `[1,C,D,H,W]` have it, and an older backend sends none. */
  channels?: ChannelPage | null;
}

/** One page of at most 16 channels. Every map is the same bounded window as the main slice; the statistics cover each whole channel. */
export interface ChannelPage {
  start: number;
  count: number;
  total: number;
  selected: number;
  shape: number[];
  rows: number;
  columns: number;
  truncated: boolean;
  maps: (number | null)[][][];
  stats: { min: number | null; max: number | null; mean: number | null }[];
}

/**
 * One weight or bias tensor of a layer, from the model the observation ran (`nodeId` is that
 * layer). A layer that shares weights reports its owner's, and says so in `owner`.
 */
export interface ParameterSnapshot extends TensorSnapshot {
  name: string;
  owner?: string;
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

/**
 * One query head of the selected branch in the overview: its softmax probabilities at the
 * selected window (at most 16 × 16), the mean normalized row entropy (0 = one key, 1 = evenly
 * spread over the keys a row may see) and the mean of each row's largest probability.
 */
export interface HeadSummary {
  head: number;
  kvHead: number;
  rows: number;
  columns: number;
  values: (number | null)[][];
  entropy: number | null;
  peak: number | null;
}

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
  /** `causal`: every key after the query's own position was set to −∞ before the softmax. */
  mask: 'none' | 'causal';
  dropout: 0;
  scoreSource: 'projected-qk';
  outputSource: 'scaled_dot_product_attention';
  tensors: Record<AttentionTensor, AttentionMatrix>;
  /** Every head of the branch; an older backend or snapshot has none. */
  overview?: HeadSummary[];
}

/**
 * One term of an explained value. `input` is where the layer's input was read (batch axis left
 * out), null where the window lies in the zero padding; `parameter` is the weight it was
 * multiplied by. Pooling has no weights: its term is x ÷ divisor (average) or x itself (max).
 */
export interface ValueTerm {
  input: number[] | null;
  x: number | null;
  parameter: number[] | null;
  w: number | null;
  term: number | null;
}

/**
 * One output element of a Linear, convolution or pooling layer, split into the measured terms
 * it is made of. `value` is the float32 output the layer produced; `recomputed` rebuilds it in
 * float64 from the same input and weights. The largest terms are listed and the rest summed.
 */
export interface ValueExplanation {
  nodeId: string;
  op: string;
  /** The output element, batch axis left out. */
  index: number[];
  rule: 'weighted-sum' | 'mean' | 'max';
  value: number | null;
  recomputed: number | null;
  bias: number | null;
  divisor: number | null;
  /** Terms counted, and window positions that fall in the padding. */
  count: number;
  padded: number;
  terms: ValueTerm[];
  rest: { count: number; sum: number | null };
  /** Convolutions only: Σ x·w per input channel, the largest first. */
  channels: { channel: number; sum: number | null }[];
  channelRest: { count: number; sum: number | null } | null;
  inputNode: string;
  inputShape: number[];
  /** A folded layer explains its last instance; a layer that shares weights names its owner. */
  instance: number;
  instances: number;
  owner?: string;
}

/**
 * One backward pass of the cross-entropy between the output logits and `target`, in eval mode.
 * `tensors` are the gradients of the captured layer outputs (a layer whose output has none, such
 * as integer IDs, is absent) and `parameters` those of the requested layer's weights and biases.
 */
export interface GradientReport {
  target: number;
  classes: number;
  loss: number | null;
  predicted: number;
  probability: number | null;
  tensors: TensorSnapshot[];
  parameters: ParameterSnapshot[];
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
  /** The requested layer's weights and biases; an older backend sends none. */
  parameters?: ParameterSnapshot[];
  /** The requested output value, split into its terms; null when none was requested. */
  explanation?: ValueExplanation | null;
  /** Gradients of the requested backward pass; null when none was requested. */
  gradient?: GradientReport | null;
}

export function tensorNumber(value: number | null) {
  if (value === null || !Number.isFinite(value)) return '非有限值';
  return value === 0 ? '0' : Math.abs(value) >= 10000 || Math.abs(value) < 0.001 ? value.toExponential(3) : Number(value.toPrecision(5)).toString();
}
