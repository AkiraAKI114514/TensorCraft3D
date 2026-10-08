import { useState } from 'react';
import type { AttentionSnapshot, AttentionTensor } from './inferenceTypes';
import { ATTENTION_TENSORS, tensorNumber } from './inferenceTypes';
import AttentionMatrix from './AttentionMatrix';
import { useI18n } from './i18n';

export default function AttentionInspector({ snapshot, transformer }: { snapshot: AttentionSnapshot; transformer: boolean }) {
  const { t } = useI18n();
  const [tensor, setTensor] = useState<AttentionTensor>('probabilities');
  const matrix = snapshot.tensors[tensor];
  return <section className="attention-observation" aria-label="Attention Inspector">
    <h3>{t('真实 Attention · B{branch} / H{head}', { branch: snapshot.branch + 1, head: snapshot.head + 1 })}</h3>
    <p className="tensor-note">{t('Query H{head} 使用 KV 组 {kvHead} / {kvHeads} · d={headDim} · 缩放 {scale} · Q 长度 {queryLength} / KV 长度 {keyLength}', { head: snapshot.head + 1, kvHead: snapshot.kvHead + 1, kvHeads: snapshot.kvHeads, headDim: snapshot.headDim, scale: tensorNumber(snapshot.scale), queryLength: snapshot.queryLength, keyLength: snapshot.keyLength })}</p>
    <p className="tensor-note">{t('同一次单样本 eval · 无 mask · dropout=0。Q/K/V、Head 和分支输出来自实际运行；分数和概率由这些 Q/K 重算，可能与融合 SDPA 有浮点差异，不替换实际计算。')}</p>
    {transformer && <p className="tensor-note">{t('Attention 分支平均不是 Transformer 最终输出；最终输出还包含残差与 FFN，请查看下方层输出。')}</p>}
    <label className="field-label">{t('内部张量')}<select aria-label={t('Attention 内部张量')} value={tensor} onChange={event => setTensor(event.target.value as AttentionTensor)}>{Object.entries(ATTENTION_TENSORS).map(([key, title]) => <option value={key} key={key}>{t(title)}</option>)}</select></label>
    <dl className="tensor-stats"><div><dt>{t('完整矩阵 Shape · ')}{matrix.dtype}</dt><dd>{`[${matrix.shape.join(', ')}]`}</dd></div><div><dt>{t('非有限元素')}</dt><dd>{matrix.nonFiniteCount}</dd></div><div><dt>{t('Mean · 全矩阵有限值')}</dt><dd>{t(tensorNumber(matrix.stats.mean))}</dd></div><div><dt>Min</dt><dd>{t(tensorNumber(matrix.stats.min))}</dd></div><div><dt>Max</dt><dd>{t(tensorNumber(matrix.stats.max))}</dd></div><div><dt>Std · {t('全矩阵总体')}</dt><dd>{t(tensorNumber(matrix.stats.std))}</dd></div></dl>
    <AttentionMatrix key={tensor} title={t(ATTENTION_TENSORS[tensor])} matrix={matrix} probability={tensor === 'probabilities'} columns={tensor === 'scores' || tensor === 'probabilities' ? 'Key' : t('特征')} rows={tensor === 'k' || tensor === 'v' ? 'Key' : 'Query'} />
  </section>;
}
