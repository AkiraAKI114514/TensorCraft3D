import { DEFAULTS, type Graph, type Op, type Params } from './types';

function sequential(name: string, specs: [Op, Params?, string?][]): Graph {
  const nodes = specs.map(([op, params, label], i) => ({ id: `layer_${i}`, name: label || `${op}_${i}`, op, params: { ...DEFAULTS[op], ...params }, position: { x: i * 210, y: 100 } }));
  return { version: 1, name, nodes, edges: nodes.slice(1).map((n, i) => ({ id: `edge_${i}`, source: nodes[i].id, target: n.id })) };
}
export const PRESETS: Record<string, () => Graph> = {
  cnn: () => sequential('Compact CNN', [['Input', {}, '图像输入'], ['Conv2d', { out_channels: 16 }, '卷积块 01'], ['BatchNorm2d', {}, '批归一化'], ['ReLU', {}, '激活'], ['MaxPool2d', {}, '下采样'], ['Conv2d', { out_channels: 32 }, '卷积块 02'], ['ReLU', {}, '激活'], ['AdaptiveAvgPool2d', {}, '全局池化'], ['Flatten', {}, '展平'], ['Linear', { out_features: 10 }, '分类器'], ['Output', {}, '分类输出']]),
  mlp: () => sequential('Feature MLP', [['Input', { shape: [1, 16] }, '特征输入'], ['Linear', { out_features: 64 }], ['ReLU'], ['Dropout'], ['Linear', { out_features: 32 }], ['GELU'], ['Linear', { out_features: 3 }, '分类器'], ['Output', {}, '分类输出']]),
  residual: () => {
    const graph = sequential('Residual CNN', [['Input'], ['Conv2d', { out_channels: 16 }], ['ReLU'], ['Conv2d', { out_channels: 16 }], ['BatchNorm2d'], ['Add', {}, '残差融合'], ['ReLU'], ['AdaptiveAvgPool2d'], ['Flatten'], ['Linear', { out_features: 10 }], ['Output']]);
    graph.edges.push({ id: 'skip', source: 'layer_2', target: 'layer_5' });
    return graph;
  },
  transformer: () => sequential('Transformer Encoder', [['Input', { shape: [1, 16, 64] }, 'Token 输入'], ['Transformer', { embed_dim: 64, num_heads: 1, ff_dim: 128 }, '多头注意力层'], ['Transformer', { embed_dim: 64, num_heads: 1, ff_dim: 128 }, '多头注意力层 02'], ['Flatten', {}, '序列展平'], ['Linear', { out_features: 10 }, '分类器'], ['Output', {}, '分类输出']]),
  mqa: () => sequential('Multi-Query Attention', [['Input', { shape: [1, 16, 64] }, 'Token 输入'], ['MultiHeadAttention', { attention_type: 'multi_query', num_heads: 4, kv_heads: 1 }, '共享 KV 注意力'], ['Flatten'], ['Linear', { out_features: 10 }], ['Output']]),
  gqa: () => sequential('Grouped-Query Attention', [['Input', { shape: [1, 16, 64] }, 'Token 输入'], ['Transformer', { attention_type: 'grouped_query', num_heads: 8, kv_heads: 2 }, '分组 KV 编码器'], ['Flatten'], ['Linear', { out_features: 10 }], ['Output']]),
  cross_attention: () => {
    const graph = sequential('Cross-Attention', [['Input', { shape: [1, 8, 64] }, 'Query 输入'], ['MultiHeadAttention', { attention_type: 'cross', num_heads: 4, kv_heads: 4 }, 'Cross-Attention'], ['Flatten'], ['Linear', { out_features: 10 }], ['Output']]);
    graph.nodes.push({ id: 'context', name: 'Context 输入', op: 'Input', params: { shape: [1, 12, 64] }, position: { x: 0, y: 320 } });
    graph.edges[0].targetPort = 'query';
    graph.edges.push({ id: 'context_cross', source: 'context', target: 'layer_1', targetPort: 'context' });
    return graph;
  },
  blank: () => sequential('Untitled model', [['Input', {}, '输入'], ['Output', {}, '输出']])
};
