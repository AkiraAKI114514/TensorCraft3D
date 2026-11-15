import { DEFAULTS, type Graph, type Op, type Params } from './types';

function sequential(name: string, specs: [Op, Params?, string?][]): Graph {
  const nodes = specs.map(([op, params, label], i) => ({ id: `layer_${i}`, name: label || `${op}_${i}`, op, params: { ...DEFAULTS[op], ...params }, position: { x: i * 210, y: 100 } }));
  return { version: 1, name, nodes, edges: nodes.slice(1).map((n, i) => ({ id: `edge_${i}`, source: nodes[i].id, target: n.id })) };
}
/** Explicit builder for graphs with branches: positions and edges are given, not inferred. */
function wired(name: string, specs: [string, Op, Params, string, number, number][], edges: [string, string][]): Graph {
  return {
    version: 1, name,
    nodes: specs.map(([id, op, params, label, x, y]) => ({ id, name: label, op, params: { ...DEFAULTS[op], ...params }, position: { x, y } })),
    edges: edges.map(([source, target], i) => ({ id: `edge_${i}`, source, target }))
  };
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
  // 命名刻意不写 DeepSeek：这个块是 GQA + 前置 RMSNorm + SwiGLU 前馈，属于 Llama/Qwen
  // 一族的解码器块。DeepSeek 的名字只给同时具备 MLA 与 top-k MoE 的结构，见下面的 deepseek。
  decoder: () => wired('Gated Decoder Block', [
    ['input', 'Input', { shape: [1, 8, 32] }, 'Token 输入', 0, 100],
    ['norm', 'RMSNorm', { normalized_shape: 32 }, '前置归一化', 210, 100],
    ['attention', 'MultiHeadAttention', { attention_type: 'grouped_query', embed_dim: 32, num_heads: 4, kv_heads: 2, dropout: 0 }, 'GQA 注意力', 420, 100],
    ['gate', 'Linear', { out_features: 64 }, '门控投影', 630, 100],
    ['value', 'Linear', { out_features: 64 }, '值投影', 630, 260],
    ['silu', 'SiLU', {}, '门控激活', 840, 100],
    ['gated', 'Multiply', {}, '逐元素门控', 1050, 180],
    ['down', 'Linear', { out_features: 32 }, '下投影', 1260, 180],
    ['flatten', 'Flatten', {}, '序列展平', 1470, 180],
    ['classifier', 'Linear', { out_features: 10 }, '分类器', 1680, 180],
    ['output', 'Output', {}, '分类输出', 1890, 180]
  ], [
    ['input', 'norm'], ['norm', 'attention'],
    ['attention', 'gate'], ['attention', 'value'],
    ['gate', 'silu'], ['silu', 'gated'], ['value', 'gated'],
    ['gated', 'down'], ['down', 'flatten'], ['flatten', 'classifier'], ['classifier', 'output']
  ]),
  // 缩小版 DeepSeek-V3：每个块都是多头潜在注意力（MLA），第一块用稠密 SwiGLU 前馈，其余块用
  // DeepSeekMoE（sigmoid 门控、top-k 路由专家加一个共享专家），与 V3 前几层保持稠密一致；投影无偏置，
  // RMSNorm 的 eps 为 1e-6。不含 MTP、FP8、专家并行、KV cache 与 YaRN；训练仍是取最后位置做分类。
  deepseek: () => {
    const norm = { normalized_shape: 128, eps: 1e-6, elementwise_affine: 1 };
    const mla = { embed_dim: 128, num_heads: 4, q_lora_rank: 64, kv_lora_rank: 32, qk_nope_head_dim: 32, qk_rope_head_dim: 16, v_head_dim: 32, rope_theta: 10000, causal: 1, eps: 1e-6 };
    const moe = { experts: 8, top_k: 2, shared_experts: 1, expert_dim: 64, score_func: 'sigmoid', route_scale: 1, groups: 1, topk_groups: 1, gate_bias: 1 };
    // The main path runs left to right; `below` places a parallel branch under the last main layer.
    type Spec = [string, Op, Params, string, number, number];
    let x = 0;
    const at = (id: string, op: Op, params: Params, label: string): Spec => { x += 210; return [id, op, params, label, x - 210, 100]; };
    const below = (id: string, op: Op, params: Params, label: string): Spec => [id, op, params, label, x - 210, 260];
    const specs: Spec[] = [
      at('tokens', 'Input', { shape: [1, 16] }, 'Token 输入'), at('embed', 'Embedding', { num_embeddings: 128, embedding_dim: 128 }, '词嵌入'),
      at('attn_norm_1', 'RMSNorm', norm, '注意力归一化 1'), at('mla_1', 'LatentAttention', mla, 'MLA 1'), at('residual_1a', 'Add', {}, '残差 1a'),
      at('ffn_norm_1', 'RMSNorm', norm, '前馈归一化 1'), at('gate_1', 'Linear', { out_features: 256, bias: 0 }, '门控投影 1'), below('up_1', 'Linear', { out_features: 256, bias: 0 }, '上投影 1'),
      at('silu_1', 'SiLU', {}, '门控激活 1'), at('swiglu_1', 'Multiply', {}, 'SwiGLU 1'), at('down_1', 'Linear', { out_features: 128, bias: 0 }, '下投影 1'), at('residual_1b', 'Add', {}, '残差 1b'),
      ...[2, 3].flatMap(block => [at(`attn_norm_${block}`, 'RMSNorm', norm, `注意力归一化 ${block}`), at(`mla_${block}`, 'LatentAttention', mla, `MLA ${block}`), at(`residual_${block}a`, 'Add', {}, `残差 ${block}a`), at(`ffn_norm_${block}`, 'RMSNorm', norm, `前馈归一化 ${block}`), at(`moe_${block}`, 'MoE', moe, `MoE ${block}`), at(`residual_${block}b`, 'Add', {}, `残差 ${block}b`)]),
      at('final_norm', 'RMSNorm', norm, '最终归一化'), at('last', 'Select', { dim: 1, index: -1 }, '取最后位置'), at('head', 'Linear', { out_features: 128, bias: 0 }, '输出头'), at('logits', 'Output', {}, 'Logits 输出')
    ];
    const block = (n: number, previous: string, ffn: [string, string][]): [string, string][] => [[previous, `attn_norm_${n}`], [`attn_norm_${n}`, `mla_${n}`], [`mla_${n}`, `residual_${n}a`], [previous, `residual_${n}a`], [`residual_${n}a`, `ffn_norm_${n}`], ...ffn, [`residual_${n}a`, `residual_${n}b`]];
    return wired('DeepSeek-V3-style Decoder', specs, [
      ['tokens', 'embed'],
      ...block(1, 'embed', [['ffn_norm_1', 'gate_1'], ['ffn_norm_1', 'up_1'], ['gate_1', 'silu_1'], ['silu_1', 'swiglu_1'], ['up_1', 'swiglu_1'], ['swiglu_1', 'down_1'], ['down_1', 'residual_1b']]),
      ...block(2, 'residual_1b', [['ffn_norm_2', 'moe_2'], ['moe_2', 'residual_2b']]),
      ...block(3, 'residual_2b', [['ffn_norm_3', 'moe_3'], ['moe_3', 'residual_3b']]),
      ['residual_3b', 'final_norm'], ['final_norm', 'last'], ['last', 'head'], ['head', 'logits']
    ]);
  },
  blank: () => sequential('Untitled model', [['Input', {}, '输入'], ['Output', {}, '输出']])
};
