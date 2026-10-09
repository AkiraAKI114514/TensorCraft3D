import { RESIDUAL_COLOR } from './flowGeometry';

export const OPS = [
  'Input', 'Output',
  'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'Linear', 'Bilinear',
  'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
  'ReLU', 'GELU', 'Sigmoid', 'Tanh', 'SiLU', 'LeakyReLU', 'ELU', 'SELU', 'Softplus', 'Softmax', 'LogSoftmax', 'PReLU', 'Hardsigmoid', 'Hardswish', 'Mish', 'Softsign', 'Identity',
  'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
  'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
  'Flatten', 'Unsqueeze', 'Squeeze', 'Slice', 'Select', 'ConstantAdd', 'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'Embedding', 'Upsample',
  'MultiHeadAttention', 'Transformer', 'Add', 'Concat', 'Group'
] as const;
export type Op = typeof OPS[number];
export type Params = Record<string, number | string | number[]>;
/** `repeat` 把一层同构块折叠成 N 个实例：形状不变，参数与激活按 N 计。缺省 1，旧项目无需迁移。 */
export const MAX_REPEAT = 1024;
export const repeatOf = (layer: { repeat?: number }) => layer.repeat ?? 1;
/** 子图实例的哨兵算子。它是节点，但没有自己的 params：结构来自 Graph.subgraphs。 */
export const GROUP_OP = 'Group';
export const MAX_SUBGRAPH_DEPTH = 16;
export interface Layer { id: string; name: string; op: Op; params: Params; position: { x: number; y: number }; repeat?: number; subgraph?: string; }
/** 子图定义。origin 只影响能否被自动解散：手动分组永不被自动识别撤销。 */
export interface SubgraphDef { id: string; name: string; nodes: Layer[]; edges: Edge[]; origin: 'auto' | 'manual'; }
export const isGroup = (layer: { op: Op }): boolean => layer.op === GROUP_OP;
export interface Edge { id: string; source: string; target: string; sourcePort?: string; targetPort?: string; }
export interface Graph { version: 1; name: string; nodes: Layer[]; edges: Edge[]; subgraphs?: Record<string, SubgraphDef>; }
export interface Diagnostic { id: string; level: 'error' | 'warning' | 'info'; code: string; message: string; nodeId?: string; }
export interface LayerInfo { input: number[][]; output: number[]; parameters: number; repeat: number; }
export interface Analysis { layers: Record<string, LayerInfo>; order: string[]; diagnostics: Diagnostic[]; parameters: number; activationBytes: number; valid: boolean; }
export interface Metric { epoch: number; trainLoss: number; valLoss: number; accuracy: number; gradNorm: number; layerGradients?: Record<string, number>; deadRelu?: Record<string, number>; source: 'demo' | 'training' | 'import'; }
export interface TrainingConfig { epochs: number; learningRate: number; batchSize: number; samples: number; device: 'auto' | 'cpu' | 'cuda'; dataset: 'synthetic' | 'csv'; csv?: string; validationFraction: number; earlyStopping: boolean; patience: number; }
const shape1d = { kernel_size: 3, stride: 1, padding: 1 };
const pool1d = { kernel_size: 2, stride: 2, padding: 0 };
export const COLORS: Record<Op, string> = {
  Input: '#7d91a8', Output: '#657788', Conv1d: '#29aa92', Conv2d: '#29aa92', Conv3d: '#29aa92', ConvTranspose1d: '#3b9c8b', ConvTranspose2d: '#3b9c8b', ConvTranspose3d: '#3b9c8b', Linear: '#e48369', Bilinear: '#e48369',
  BatchNorm1d: '#bc8cc8', BatchNorm2d: '#bc8cc8', BatchNorm3d: '#bc8cc8', LayerNorm: '#bc8cc8', GroupNorm: '#bc8cc8', InstanceNorm1d: '#bc8cc8', InstanceNorm2d: '#bc8cc8', InstanceNorm3d: '#bc8cc8',
  ReLU: '#e8b84b', GELU: '#e8b84b', Sigmoid: '#e8b84b', Tanh: '#e8b84b', SiLU: '#e8b84b', LeakyReLU: '#e8b84b', ELU: '#e8b84b', SELU: '#e8b84b', Softplus: '#e8b84b', Softmax: '#e8b84b', LogSoftmax: '#e8b84b', PReLU: '#e8b84b', Hardsigmoid: '#e8b84b', Hardswish: '#e8b84b', Mish: '#e8b84b', Softsign: '#e8b84b', Identity: '#8e97a7',
  MaxPool1d: '#5299cf', MaxPool2d: '#5299cf', MaxPool3d: '#5299cf', AvgPool1d: '#5299cf', AvgPool2d: '#5299cf', AvgPool3d: '#5299cf', AdaptiveAvgPool1d: '#5299cf', AdaptiveAvgPool2d: '#5299cf', AdaptiveAvgPool3d: '#5299cf', AdaptiveMaxPool1d: '#5299cf', AdaptiveMaxPool2d: '#5299cf', AdaptiveMaxPool3d: '#5299cf',
  Flatten: '#8e97a7', Unsqueeze: '#8e97a7', Squeeze: '#8e97a7', Slice: '#8e97a7', Select: '#8e97a7', ConstantAdd: '#5299cf', Dropout: '#bc8cc8', Dropout1d: '#bc8cc8', Dropout2d: '#bc8cc8', Dropout3d: '#bc8cc8', AlphaDropout: '#bc8cc8', Embedding: '#d073a8', Upsample: '#5299cf',
  MultiHeadAttention: '#3c9bb4', Transformer: '#477eae', Add: RESIDUAL_COLOR, Concat: '#d073a8', Group: '#6b8cae'
};
export const DEFAULTS: Record<Op, Params> = {
  Input: { shape: [1, 3, 32, 32] }, Output: {},
  Conv1d: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 }, Conv2d: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 }, Conv3d: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 }, ConvTranspose1d: { out_channels: 16, kernel_size: 4, stride: 2, padding: 1 }, ConvTranspose2d: { out_channels: 16, kernel_size: 4, stride: 2, padding: 1 }, ConvTranspose3d: { out_channels: 16, kernel_size: 4, stride: 2, padding: 1 },
  Linear: { out_features: 10 }, Bilinear: { in2_features: 16, out_features: 10 },
  BatchNorm1d: {}, BatchNorm2d: {}, BatchNorm3d: {}, LayerNorm: { normalized_shape: 64, eps: 1e-5, elementwise_affine: 1 }, GroupNorm: { num_groups: 1 }, InstanceNorm1d: {}, InstanceNorm2d: {}, InstanceNorm3d: {},
  ReLU: {}, GELU: {}, Sigmoid: {}, Tanh: {}, SiLU: {}, LeakyReLU: { negative_slope: 0.01 }, ELU: { alpha: 1 }, SELU: {}, Softplus: { beta: 1, threshold: 20 }, Softmax: { dim: -1 }, LogSoftmax: { dim: -1 }, PReLU: { num_parameters: 1, init: 0.25 }, Hardsigmoid: {}, Hardswish: {}, Mish: {}, Softsign: {}, Identity: {},
  MaxPool1d: pool1d, MaxPool2d: { ...pool1d }, MaxPool3d: { ...pool1d }, AvgPool1d: pool1d, AvgPool2d: { ...pool1d }, AvgPool3d: { ...pool1d },
  AdaptiveAvgPool1d: { output_size: 1 }, AdaptiveAvgPool2d: { output_size: 1 }, AdaptiveAvgPool3d: { output_size: 1 }, AdaptiveMaxPool1d: { output_size: 1 }, AdaptiveMaxPool2d: { output_size: 1 }, AdaptiveMaxPool3d: { output_size: 1 },
  Flatten: {}, Unsqueeze: { dim: 0 }, Squeeze: { dim: 'all' }, Slice: { dim: 1, start: 'none', end: 'none', step: 1 }, Select: { dim: 1, index: -1 }, ConstantAdd: { shape: [1], values: [0] }, Dropout: { p: 0.3 }, Dropout1d: { p: 0.3 }, Dropout2d: { p: 0.3 }, Dropout3d: { p: 0.3 }, AlphaDropout: { p: 0.3 }, Embedding: { num_embeddings: 100, embedding_dim: 32 }, Upsample: { scale_factor: 2, mode: 'nearest' },
  MultiHeadAttention: { attention_type: 'self', embed_dim: 64, num_heads: 1, kv_heads: 1, branches: 1, dropout: 0.1 }, Transformer: { attention_type: 'self', embed_dim: 64, num_heads: 1, kv_heads: 1, branches: 1, ff_dim: 128, dropout: 0.1 }, Add: {}, Concat: { dim: 1 }, Group: {}
};
