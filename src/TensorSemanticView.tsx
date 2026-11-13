import { useState } from 'react';
import AttentionMatrix, { matrixStep } from './AttentionMatrix';
import { tensorNumber, type AttentionMatrix as Matrix, type ChannelPage, type TensorSnapshot } from './inferenceTypes';
import { useI18n } from './i18n';
import './TensorSemanticView.css';

export type SemanticKind = 'feature-map' | 'volume' | 'sequence' | 'vector';

/**
 * 张量的秩决定它该被看成什么：`[1,C,H,W]` 是特征图，`[1,C,D,H,W]` 是带深度的特征图，
 * `[1,S,E]` 是位置 × 特征的序列，`[1,F]` 是特征向量。整数张量（Embedding 的输入 ID）
 * 的数值是类别而不是幅度，给它上色只会误导，所以没有语义视图。
 */
export function semanticKind(snapshot: TensorSnapshot): SemanticKind | null {
  if (!snapshot.dtype.startsWith('float')) return null;
  const rank = snapshot.shape.length;
  return rank === 4 ? 'feature-map' : rank === 5 ? 'volume' : rank === 3 ? 'sequence' : rank === 2 ? 'vector' : null;
}

function ChannelGrid({ page, limit, disabled, rerun, onSelect }: { page: ChannelPage; limit: number; disabled: boolean; rerun: boolean; onSelect: (channel: number) => void }) {
  const { t } = useI18n();
  const [active, setActive] = useState<number | null>(null);
  const describe = (index: number) => t('通道 {channel}：均值 {mean}，范围 {min} – {max}', { channel: page.start + index, mean: t(tensorNumber(page.stats[index].mean)), min: t(tensorNumber(page.stats[index].min)), max: t(tensorNumber(page.stats[index].max)) });
  return <figure className="attention-matrix channel-grid">
    <figcaption>{t('通道网格 · 通道 {start}–{end} / 共 {total}', { start: page.start, end: page.start + page.count - 1, total: page.total })}</figcaption>
    <p className="attention-caption">{rerun ? t('每格是该通道同一窗口的实测值，与上方特征图共用色阶；统计覆盖整个通道。点选通道会改动切片，需要重新运行。') : t('每格是该通道同一窗口的实测值，与上方特征图共用色阶；统计覆盖整个通道。点选通道直接查看它的窗口。')}</p>
    <div className="channel-grid-items" role="group" aria-label={t('通道网格')}>{page.maps.map((map, index) => {
      const channel = page.start + index, selected = channel === page.selected;
      // 选中的通道再点一次不应让快照失效，所以它只是一个只读的当前项。
      return <button type="button" key={channel} className={selected ? 'active' : ''} aria-pressed={selected} aria-label={describe(index)} title={describe(index)} disabled={disabled} onClick={() => { if (!selected) onSelect(channel); }} onMouseEnter={() => setActive(index)} onMouseLeave={() => setActive(null)} onFocus={() => setActive(index)} onBlur={() => setActive(null)}>
        <svg viewBox={`0 0 ${page.columns} ${page.rows}`} aria-hidden="true">{map.map((row, r) => row.map((value, c) => {
          const step = matrixStep(value, limit, false), sign = value !== null && value < 0 ? 'negative' : 'positive';
          return <rect key={`${r}-${c}`} x={c} y={r} width="1.03" height="1.03" className={step === null ? 'matrix-nonfinite' : value === 0 ? 'matrix-neutral' : `matrix-${sign}-${step}`} />;
        }))}</svg>
        <span>{channel}</span>
      </button>;
    })}</div>
    <p className="tensor-hover" role="status">{active === null ? t('悬停或聚焦通道查看整个通道的统计') : describe(active)}</p>
    <details><summary>{t('通道统计表 · 整个通道')}</summary><div className="tensor-table-scroll"><table className="tensor-slice"><thead><tr><th>{t('通道')}</th><th>{t('均值')}</th><th>Min</th><th>Max</th></tr></thead><tbody>{page.stats.map((stats, index) => <tr key={index}><th>{page.start + index}</th><td>{t(tensorNumber(stats.mean))}</td><td>{t(tensorNumber(stats.min))}</td><td>{t(tensorNumber(stats.max))}</td></tr>)}</tbody></table></div></details>
  </figure>;
}

/**
 * 同一份有界切片的第二种读法：数值表回答「这个元素是多少」，热力图回答「形状长什么样」。
 * 它不请求任何额外数据——热力图就是已经返回的那个 16 × 16 窗口，通道网格是后端随快照
 * 一起返回的同一窗口在相邻通道上的取值。
 */
/** `rerun` is false when the channel grid only switches between maps already in hand (an offline snapshot). */
export default function TensorSemanticView({ snapshot, disabled, rerun = true, onSelectChannel }: { snapshot: TensorSnapshot; disabled: boolean; rerun?: boolean; onSelectChannel: (channel: number) => void }) {
  const { t } = useI18n();
  const kind = semanticKind(snapshot);
  if (!kind) return null;
  const [, channel, depth] = snapshot.slice.indices;
  const spatial = kind === 'feature-map' || kind === 'volume';
  const title = kind === 'feature-map' ? t('特征图 · 通道 {channel}', { channel }) : kind === 'volume' ? t('特征图 · 通道 {channel} · 深度 {depth}', { channel, depth }) : kind === 'sequence' ? t('序列 · 位置 × 特征') : t('特征向量');
  const note = spatial ? t('固定前置轴后，行、列对应特征图的高和宽；色阶取整个张量的对称范围，所以不同通道可以直接比较。') : kind === 'sequence' ? t('行是序列位置，列是特征维度。位置只是序号，不带分词信息。') : t('单个样本的特征向量，列是特征维度。');
  // 热力图组件按「完整矩阵 + 窗口」描述数据；切片的完整矩阵就是末尾两维。
  const shape = snapshot.slice.shape.length === 2 ? snapshot.slice.shape : [1, snapshot.slice.shape[0] ?? snapshot.slice.columns];
  const matrix: Matrix = { ...snapshot, shape, rowStart: 0, columnStart: 0 };
  const limit = Math.max(Math.abs(snapshot.stats.min ?? 0), Math.abs(snapshot.stats.max ?? 0));
  return <div className="inspector-section tensor-semantic"><h3>{t('语义视图')}<span>{t('最多 16 × 16')}</span></h3>
    <p className="tensor-note">{note}</p>
    <AttentionMatrix matrix={matrix} title={title} rows={kind === 'sequence' ? '位置' : '行'} columns={spatial ? '列' : '特征'} />
    {spatial && snapshot.channels && <ChannelGrid page={snapshot.channels} limit={limit} disabled={disabled} rerun={rerun} onSelect={onSelectChannel} />}
  </div>;
}
