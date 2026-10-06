import type { Graph, Op, Params } from '../src/types';

export function imageTestGraph(): Graph {
  const encoder: Params = { embed_dim: 64, num_heads: 4, kv_heads: 4, ff_dim: 128, dropout: 0, norm_first: 1, activation: 'gelu' };
  const specs: [string, Op, Params, string?][] = [
    ['climate', 'Input', { shape: [64, 6, 101] }], ['canopy', 'Input', { shape: [64, 6, 69] }],
    ['climate_projection', 'Linear', { out_features: 64 }], ['climate_norm', 'LayerNorm', { normalized_shape: 64 }],
    ['canopy_projection', 'Linear', { out_features: 64 }], ['canopy_norm', 'LayerNorm', { normalized_shape: 64 }],
    ['climate_encoder_0', 'Transformer', encoder, 'self.climate_encoder.0'], ['climate_encoder_1', 'Transformer', encoder, 'self.climate_encoder.1'],
    ['canopy_encoder_0', 'Transformer', encoder, 'self.canopy_encoder.0'], ['canopy_encoder_1', 'Transformer', encoder, 'self.canopy_encoder.1'],
    ['cross_attention', 'MultiHeadAttention', { embed_dim: 64, num_heads: 4, kv_heads: 4, attention_type: 'cross', dropout: 0 }],
    ['cross_add', 'Add', {}], ['cross_norm', 'LayerNorm', { normalized_shape: 64 }],
    ['ffn_in', 'Linear', { out_features: 128 }, 'self.fusion_ffn.0'], ['gelu', 'GELU', {}], ['dropout', 'Dropout', { p: 0 }],
    ['ffn_out', 'Linear', { out_features: 64 }, 'self.fusion_ffn.3'], ['fusion_add', 'Add', {}], ['fusion_norm', 'LayerNorm', { normalized_shape: 64 }],
    ['head_dropout', 'Dropout', { p: 0 }], ['head', 'Linear', { out_features: 1 }, 'self.regression_head.1'], ['squeeze', 'Squeeze', { dim: -1 }], ['output', 'Output', {}],
  ];
  const edges: [string, string, string?][] = [
    ['climate', 'climate_projection'], ['climate_projection', 'climate_norm'], ['climate_norm', 'climate_encoder_0'], ['climate_encoder_0', 'climate_encoder_1'],
    ['canopy', 'canopy_projection'], ['canopy_projection', 'canopy_norm'], ['canopy_norm', 'canopy_encoder_0'], ['canopy_encoder_0', 'canopy_encoder_1'],
    ['climate_encoder_1', 'cross_attention', 'query'], ['canopy_encoder_1', 'cross_attention', 'context'],
    ['climate_encoder_1', 'cross_add'], ['cross_attention', 'cross_add'], ['cross_add', 'cross_norm'],
    ['cross_norm', 'ffn_in'], ['ffn_in', 'gelu'], ['gelu', 'dropout'], ['dropout', 'ffn_out'], ['ffn_out', 'fusion_add'], ['cross_norm', 'fusion_add'],
    ['fusion_add', 'fusion_norm'], ['fusion_norm', 'head_dropout'], ['head_dropout', 'head'], ['head', 'squeeze'], ['squeeze', 'output'],
  ];
  return { version: 1, name: 'DualBranchCrossAttentionTransformer', nodes: specs.map(([id, op, params, name], index) => ({ id, op, params: { ...params }, name: name ?? `self.${id}`, position: { x: index * 210, y: 100 } })),
    edges: edges.map(([source, target, targetPort], index) => ({ id: `connection_${index}`, source, target, ...(targetPort ? { targetPort } : {}) })) };
}
