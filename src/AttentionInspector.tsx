import { useState } from 'react';
import type { AttentionSnapshot, AttentionTensor, HeadSummary } from './inferenceTypes';
import { ATTENTION_TENSORS, tensorNumber } from './inferenceTypes';
import AttentionMatrix, { matrixStep } from './AttentionMatrix';
import { useI18n } from './i18n';

/**
 * Every query head of the branch side by side, the way BertViz's head view lets you compare
 * them: each tile is that head's probabilities at the same window, on the fixed 0–1 scale, with
 * its entropy and peak. Choosing a tile picks the head the detail below shows; live, that takes
 * a rerun, and a snapshot only holds the detail of the head it captured.
 */
function HeadOverview({ heads, selected, branch, onSelect }: { heads: HeadSummary[]; selected: number; branch: number; onSelect?: (head: number) => void }) {
  const { t } = useI18n();
  const [active, setActive] = useState<number | null>(null);
  const share = (value: number | null) => value === null ? '—' : value.toFixed(2);
  const describe = (entry: HeadSummary) => t('H{head}：平均熵 {entropy}，平均最大概率 {peak}', { head: entry.head + 1, entropy: share(entry.entropy), peak: share(entry.peak) });
  return <figure className="attention-matrix channel-grid head-overview">
    <figcaption>{t('全部 Head · B{branch} · 概率窗口', { branch: branch + 1 })}</figcaption>
    <p className="attention-caption">{onSelect ? t('每格是一个 Query head 的 softmax 概率，窗口与下方相同，色阶固定 0 到 1。熵越低，注意力越集中。点选会切换下方详情的 head，需要重新运行。') : t('每格是一个 Query head 的 softmax 概率，窗口与下方相同，色阶固定 0 到 1。熵越低，注意力越集中。快照只含采集时那个 head 的详情。')}</p>
    <div className="channel-grid-items" role="group" aria-label={t('全部 Head')}>{heads.map(entry => {
      const current = entry.head === selected;
      // Read-only tiles stay focusable, so the keyboard still reaches each head's entropy and peak.
      return <button type="button" key={entry.head} className={current ? 'active' : ''} aria-pressed={current} aria-disabled={onSelect ? undefined : true} aria-label={describe(entry)} title={describe(entry)} onClick={() => { if (!current) onSelect?.(entry.head); }} onMouseEnter={() => setActive(entry.head)} onMouseLeave={() => setActive(null)} onFocus={() => setActive(entry.head)} onBlur={() => setActive(null)}>
        <svg viewBox={`0 0 ${Math.max(1, entry.columns)} ${Math.max(1, entry.rows)}`} aria-hidden="true">{entry.values.map((row, r) => row.map((value, c) => {
          const step = matrixStep(value, 1, true);
          return <rect key={`${r}-${c}`} x={c} y={r} width="1.03" height="1.03" className={step === null ? 'matrix-nonfinite' : value === 0 ? 'matrix-neutral' : `matrix-positive-${step}`} />;
        }))}</svg>
        <span>H{entry.head + 1}</span>
      </button>;
    })}</div>
    <p className="tensor-hover" role="status">{active === null ? t('悬停或聚焦 head 查看它的熵和最大概率') : describe(heads.find(entry => entry.head === active)!)}</p>
    <details><summary>{t('Head 统计表 · 完整矩阵')}</summary><div className="tensor-table-scroll"><table className="tensor-slice"><thead><tr><th>Head</th><th>{t('KV 组')}</th><th>{t('平均熵')}</th><th>{t('平均最大概率')}</th></tr></thead><tbody>{heads.map(entry => <tr key={entry.head}><th>H{entry.head + 1}</th><td>{entry.kvHead + 1}</td><td>{share(entry.entropy)}</td><td>{share(entry.peak)}</td></tr>)}</tbody></table></div></details>
  </figure>;
}

export default function AttentionInspector({ snapshot, transformer, onSelectHead }: { snapshot: AttentionSnapshot; transformer: boolean; onSelectHead?: (head: number) => void }) {
  const { t } = useI18n();
  const [tensor, setTensor] = useState<AttentionTensor>('probabilities');
  const matrix = snapshot.tensors[tensor];
  return <section className="attention-observation" aria-label="Attention Inspector">
    <h3>{t('真实 Attention · B{branch} / H{head}', { branch: snapshot.branch + 1, head: snapshot.head + 1 })}</h3>
    {snapshot.overview && snapshot.overview.length > 1 && <HeadOverview heads={snapshot.overview} selected={snapshot.head} branch={snapshot.branch} onSelect={onSelectHead} />}
    <p className="tensor-note">{t('Query H{head} 使用 KV 组 {kvHead} / {kvHeads} · d={headDim} · 缩放 {scale} · Q 长度 {queryLength} / KV 长度 {keyLength}', { head: snapshot.head + 1, kvHead: snapshot.kvHead + 1, kvHeads: snapshot.kvHeads, headDim: snapshot.headDim, scale: tensorNumber(snapshot.scale), queryLength: snapshot.queryLength, keyLength: snapshot.keyLength })}</p>
    <p className="tensor-note">{snapshot.mask === 'causal'
      ? t('同一次单样本 eval · causal 掩码 · dropout=0。Q/K/V、Head 和分支输出来自实际运行；分数是掩码前的 QKᵀ/√d，概率由屏蔽当前位置之后的 Key 再做 softmax 重算，可能与融合 SDPA 有浮点差异，不替换实际计算。')
      : t('同一次单样本 eval · 无 mask · dropout=0。Q/K/V、Head 和分支输出来自实际运行；分数和概率由这些 Q/K 重算，可能与融合 SDPA 有浮点差异，不替换实际计算。')}</p>
    {transformer && <p className="tensor-note">{t('Attention 分支平均不是 Transformer 最终输出；最终输出还包含残差与 FFN，请查看下方层输出。')}</p>}
    <label className="field-label">{t('内部张量')}<select aria-label={t('Attention 内部张量')} value={tensor} onChange={event => setTensor(event.target.value as AttentionTensor)}>{Object.entries(ATTENTION_TENSORS).map(([key, title]) => <option value={key} key={key}>{t(title)}</option>)}</select></label>
    <dl className="tensor-stats"><div><dt>{t('完整矩阵 Shape · ')}{matrix.dtype}</dt><dd>{`[${matrix.shape.join(', ')}]`}</dd></div><div><dt>{t('非有限元素')}</dt><dd>{matrix.nonFiniteCount}</dd></div><div><dt>{t('Mean · 全矩阵有限值')}</dt><dd>{t(tensorNumber(matrix.stats.mean))}</dd></div><div><dt>Min</dt><dd>{t(tensorNumber(matrix.stats.min))}</dd></div><div><dt>Max</dt><dd>{t(tensorNumber(matrix.stats.max))}</dd></div><div><dt>Std · {t('全矩阵总体')}</dt><dd>{t(tensorNumber(matrix.stats.std))}</dd></div></dl>
    <AttentionMatrix key={tensor} title={t(ATTENTION_TENSORS[tensor])} matrix={matrix} probability={tensor === 'probabilities'} columns={tensor === 'scores' || tensor === 'probabilities' ? 'Key' : t('特征')} rows={tensor === 'k' || tensor === 'v' ? 'Key' : 'Query'} masked={snapshot.mask === 'causal' && (tensor === 'scores' || tensor === 'probabilities') ? (row, column) => column > row : undefined} />
  </section>;
}
