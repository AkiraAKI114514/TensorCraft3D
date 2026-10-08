import { useState } from 'react';
import type { AttentionSnapshot, AttentionTensor } from './inferenceTypes';
import { ATTENTION_TENSORS, tensorNumber } from './inferenceTypes';
import AttentionMatrix from './AttentionMatrix';

export default function AttentionInspector({ snapshot, transformer }: { snapshot: AttentionSnapshot; transformer: boolean }) {
  const [tensor, setTensor] = useState<AttentionTensor>('probabilities');
  const matrix = snapshot.tensors[tensor];
  return <section className="attention-observation" aria-label="Attention Inspector">
    <h3>真实 Attention · B{snapshot.branch + 1} / H{snapshot.head + 1}</h3>
    <p className="tensor-note">Query H{snapshot.head + 1} 使用 KV 组 {snapshot.kvHead + 1} / {snapshot.kvHeads} · d={snapshot.headDim} · 缩放 {tensorNumber(snapshot.scale)} · Q 长度 {snapshot.queryLength} / KV 长度 {snapshot.keyLength}</p>
    <p className="tensor-note">同一次单样本 eval · 无 mask · dropout=0。Q/K/V、Head 和分支输出来自实际运行；分数和概率由这些 Q/K 重算，可能与融合 SDPA 有浮点差异，不替换实际计算。</p>
    {transformer && <p className="tensor-note">Attention 分支平均不是 Transformer 最终输出；最终输出还包含残差与 FFN，请查看下方层输出。</p>}
    <label className="field-label">内部张量<select aria-label="Attention 内部张量" value={tensor} onChange={event => setTensor(event.target.value as AttentionTensor)}>{Object.entries(ATTENTION_TENSORS).map(([key, title]) => <option value={key} key={key}>{title}</option>)}</select></label>
    <dl className="tensor-stats"><div><dt>完整矩阵 Shape · {matrix.dtype}</dt><dd>{`[${matrix.shape.join(', ')}]`}</dd></div><div><dt>非有限元素</dt><dd>{matrix.nonFiniteCount}</dd></div><div><dt>Mean · 全矩阵有限值</dt><dd>{tensorNumber(matrix.stats.mean)}</dd></div><div><dt>Min</dt><dd>{tensorNumber(matrix.stats.min)}</dd></div><div><dt>Max</dt><dd>{tensorNumber(matrix.stats.max)}</dd></div><div><dt>Std · 全矩阵总体</dt><dd>{tensorNumber(matrix.stats.std)}</dd></div></dl>
    <AttentionMatrix key={tensor} title={ATTENTION_TENSORS[tensor]} matrix={matrix} probability={tensor === 'probabilities'} columns={tensor === 'scores' || tensor === 'probabilities' ? 'Key' : '特征'} rows={tensor === 'k' || tensor === 'v' ? 'Key' : 'Query'} />
  </section>;
}
